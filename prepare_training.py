import os
import datetime
import warnings
from random import randint
warnings.filterwarnings(action='ignore')
import numpy as np
from PIL import Image\
#from tqdm import tqdm
from scipy.interpolate import griddata as interp_grid
from scipy.ndimage import minimum_filter, maximum_filter
from PIL import Image
from scipy.ndimage import map_coordinates
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = './ffmpeg'
import requests
from io import BytesIO
from typing import Dict, Union
import subprocess
import torch
import torch.nn.functional as F
import boto3
from botocore.exceptions import NoCredentialsError
from scipy.spatial.transform import Rotation as RR
from einops import rearrange
import requests
from PIL import Image, ImageDraw, ImageFilter
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from diffusers import (
    StableDiffusionXLInpaintPipeline,
    EulerDiscreteScheduler,
)
import sys
original_sys_path = sys.path.copy()
sys.path.append('/workspace/gaussian-opacity-field/ip_adapter')
from ip_adapter import IPAdapterXL
sys.path = original_sys_path

import os
print("Current working directory:", os.getcwd())

from utils.train_utils import get_pcdGenPoses,panorama_to_plane,generate_and_extract_full_sphere_poses,prepare_top_and_bottom_mask,calculate_white_ratio,expand_white_pixels
from utils.train_utils import CameraParams, GSParams

import json
import copy
from argparse import ArgumentParser



processor = BlipProcessor.from_pretrained('/workspace/gaussian-opacity-field/blip_models')
blip_model = BlipForConditionalGeneration.from_pretrained('/workspace/gaussian-opacity-field/blip_models').to('cuda')


### ready the inpaint pipeline
base_model_path = "/workspace/gaussian-opacity-field/inpaint_base_models"
#base_model_path = "SG161222/RealVisXL_V4.0_Lightning"
image_encoder_path = "/workspace/gaussian-opacity-field/sdxl_models/image_encoder"
ip_ckpt = "/workspace/gaussian-opacity-field/sdxl_models/ip-adapter_sdxl.bin"
device = "cuda"
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.enable_vae_tiling()
#pipe.scheduler = EulerDiscreteScheduler.from_config(
#    pipe.scheduler.config, timestep_spacing="trailing", prediction_type="epsilon"
#)

# load ip-adapter
# target_blocks=["block"] for original IP-Adapter
# target_blocks=["up_blocks.0.attentions.1"] for style blocks only
target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=target_blocks)

opt = GSParams()
cam = CameraParams()
save_dir='./'
for_gradio = False
root = 'outputs'

timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')

bg_color = [1, 1, 1] if opt.white_background else [0, 0, 0]

get_kernel = lambda p: torch.ones(1, 1, p * 2 + 1, p * 2 + 1).to('cuda')
t2np = lambda x: (x[0].permute(1, 2, 0).clamp_(0, 1) * 255.0).to(torch.uint8).detach().cpu().numpy()
np2t = lambda x: (torch.as_tensor(x).to(torch.float32).permute(2, 0, 1) / 255.0)[None, ...].to('cuda')
pad_mask = lambda x, padamount=1: t2np(
    F.conv2d(np2t(x[..., None]), get_kernel(padamount), padding=padamount))[..., 0].astype(bool)


n_horizontal = 10  # Number of horizontal divisions
n_vertical = 5    # Number of vertical divisions
yaws, pitches, render_poses = generate_and_extract_full_sphere_poses(n_horizontal, n_vertical)
#print("Yaws:", yaws)
#print("Pitches:", pitches)
#print("Render Poses shape:", render_poses.shape)

pitches = np.array(pitches)+90
yaws = -np.array(yaws)
#print("Yaws:", yaws)
#print("Pitches:", pitches)

background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
d_model = torch.hub.load('./ZoeDepth', 'ZoeD_N', source='local', pretrained=True).to('cuda')
controlnet = None
lama = None


def d(im):
    return d_model.infer_pil(im)

H, W, K = 1024, 1024, cam.K
#FOV = 120
FOV = 110
focal = (0.5 * W) / np.tan(np.radians(FOV) / 2)
FOCAL = [focal,focal]
K = np.array([
            [FOCAL[0], 0., W/2],
            [0.,FOCAL[1], H/2],
            [0.,            0.,       1.],
        ]).astype(np.float32)
fov = (2*np.arctan(W / (2*FOCAL[0])), 2*np.arctan(H / (2*FOCAL[1])))

image_360_path = "zcabnzh__00000_.png"
bucket_name = 'splatminiworlds'

def save_data(data, base_path):
    # 创建数据的深拷贝以避免修改原始数据
    data_copy = copy.deepcopy(data)
    
    # 处理图像并保存
    if 'frames' in data_copy:
        for i, frame in enumerate(data_copy['frames']):
            img_path = f'{base_path}/image_{i}.jpg'
            frame['image'].save(img_path)
            frame['image'] = img_path  # 存储图像文件的路径

    # 处理数组数据并保存
    if 'pcd_points' in data_copy:
        np.save(f'{base_path}/pcd_points.npy', data_copy['pcd_points'])
        data_copy['pcd_points'] = f'{base_path}/pcd_points.npy'
    
    if 'pcd_colors' in data_copy:
        np.save(f'{base_path}/pcd_colors.npy', data_copy['pcd_colors'])
        data_copy['pcd_colors'] = f'{base_path}/pcd_colors.npy'

    # 保存所有数据（包括数值型数据）为 JSON
    with open(f'{base_path}/data.json', 'w') as fp:
        json.dump(data_copy, fp, indent=4)


def infer(image_360_path = "zcabnzh__00000_.png"):
    if os.path.exists(image_360_path):
        print("File exists")
    else:
        print("File does not exist at", image_360_path)
        image_360_path = "zcabnzh__00000_.png"
    image_360 = Image.open(image_360_path)
    width, height = image_360.size
    ###
    ### for inpainting style
    image_curr = panorama_to_plane(image_360, FOV, (H, W), yaws[0], pitches[0])
    #image_curr.show()
    
    style_ref_images = []
    style_ref_image = image_curr.copy()
    style_ref_image.resize((512, 512))
    style_ref_images.append(style_ref_image)
    w_in, h_in = image_curr.size
    depth_curr = d(image_curr)
    center_depth = np.mean(depth_curr[h_in//2-10:h_in//2+10, w_in//2-10:w_in//2+10])
    
    x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy') # pixels
    edgeN = 2
    
    edgemask = np.ones((H-2*edgeN, W-2*edgeN))
    edgemask = np.pad(edgemask, ((edgeN,edgeN),(edgeN,edgeN)))
    
    R0, T0 = render_poses[0,:3,:3], render_poses[0,:3,3:4]
    
    pts_coord_cam = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))
    
    new_pts_coord_world2 = (np.linalg.inv(R0).dot(pts_coord_cam) - np.linalg.inv(R0).dot(T0)).astype(np.float32) ## new_pts_coord_world2
    
    new_pts_colors2 = (np.array(image_curr).reshape(-1,3).astype(np.float32)/255.) ## new_pts_colors2
    pts_coord_world, pts_colors = new_pts_coord_world2.copy(), new_pts_colors2.copy()
    iterable_dream = range(1, len(render_poses))
    top_bottom_mask = prepare_top_and_bottom_mask(4)
    traindata = {
            'camera_angle_x': fov[0],
            'W': W,
            'H': H,
            'pcd_points': [],
            'pcd_colors': [],
            'frames': [],
            }


    for i in iterable_dream:
        print(f'{i} / {iterable_dream}')
        R, T = render_poses[i,:3,:3], render_poses[i,:3,3:4]
        pts_coord_cam2 = R.dot(pts_coord_world) + T  ### Same with c2w*world_coord (in homogeneous space)
        pixel_coord_cam2 = np.matmul(K, pts_coord_cam2) 
        valid_idx = np.where(np.logical_and.reduce((pixel_coord_cam2[2]>0, 
                                                                pixel_coord_cam2[0]/pixel_coord_cam2[2]>=0, 
                                                                pixel_coord_cam2[0]/pixel_coord_cam2[2]<=W-1, 
                                                                pixel_coord_cam2[1]/pixel_coord_cam2[2]>=0, 
                                                                pixel_coord_cam2[1]/pixel_coord_cam2[2]<=H-1)))[0]
        
        pixel_coord_cam2 = pixel_coord_cam2[:2, valid_idx]/pixel_coord_cam2[-1:, valid_idx]
        round_coord_cam2 = np.round(pixel_coord_cam2).astype(np.int32)
        image_curr = panorama_to_plane(image_360, FOV, (H, W), yaws[i], pitches[i])
        #image_curr.show()
    
        if i in iterable_dream[-2:]:
            text = ""
            cropped_image = image_curr.crop((312, 312, 712, 712))
            #cropped_image.show()
            inputs = processor(cropped_image, text, return_tensors="pt").to("cuda")
            out = blip_model.generate(**inputs)
            inpaint_prompt = processor.decode(out[0], skip_special_tokens=True)
            print(inpaint_prompt)
    
            if i == iterable_dream[-1]:
                inpaint_prompt += ", ground, Land, terrain"
            else:
                inpaint_prompt += ", (sky:1.1)"
    
    
            #images = ip_model.generate(pil_image=style_ref_image,
                                #prompt=inpaint_prompt,
                                #negative_prompt= "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                                #scale=2.0,
                                #guidance_scale=8,
                                #num_samples=1,
                                #num_inference_steps=25, 
                                #image=image_curr,
                                #mask_image=top_bottom_mask,
                                #strength=0.99
                                #)
    
    
            images = ip_model.generate(pil_image=style_ref_image,
                                prompt=inpaint_prompt,
                                negative_prompt= "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                                scale=2.0,
                                guidance_scale=7,
                                num_samples=1,
                                num_inference_steps=24, 
                                image=image_curr,
                                mask_image=top_bottom_mask,
                                strength=0.5
                                )
    
            #images[0].save("result.png")
            image_curr = images[0]
            #image_curr.show()
    
        style_ref_image = image_curr.copy()
        style_ref_image.resize((512, 512))
        style_ref_images.append(style_ref_image)
        
        x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    
        grid = np.stack((x,y), axis=-1).reshape(-1,2)
        image2 = interp_grid(pixel_coord_cam2.transpose(1,0), pts_colors[valid_idx], grid, method='linear', fill_value=0).reshape(H,W,3)
        image2 = edgemask[...,None]*image2 + (1-edgemask[...,None])*np.pad(image2[1:-1,1:-1], ((1,1),(1,1),(0,0)), mode='edge')
        
        round_mask2 = np.zeros((H,W), dtype=np.float32)
        round_mask2[round_coord_cam2[1], round_coord_cam2[0]] = 1
        
        round_mask2 = maximum_filter(round_mask2, size=(9,9), axes=(0,1))
        image2 = round_mask2[...,None]*image2 + (1-round_mask2[...,None])*(-1)
        
        mask2 = minimum_filter((image2.sum(-1)!=-3)*1, size=(11,11), axes=(0,1))
        image2 = mask2[...,None]*image2 + (1-mask2[...,None])*0
        
        mask_hf = np.abs(mask2[:H-1, :W-1] - mask2[1:, :W-1]) + np.abs(mask2[:H-1, :W-1] - mask2[:H-1, 1:])
        mask_hf = np.pad(mask_hf, ((0,1), (0,1)), 'edge')
        mask_hf = np.where(mask_hf < 0.3, 0, 1)
        border_valid_idx = np.where(mask_hf[round_coord_cam2[1], round_coord_cam2[0]] == 1)[0]
      
        depth_curr = d(image_curr)
        #visualize_depth(depth_curr*50)
        
        t_z2 = torch.tensor(depth_curr)
        sc = torch.ones(1).float().requires_grad_(True)
        optimizer = torch.optim.Adam(params=[sc], lr=0.001)
    
        for idx in range(100):
            trans3d = torch.tensor([[sc,0,0,0], [0,sc,0,0], [0,0,sc,0], [0,0,0,1]]).requires_grad_(True)
            coord_cam2 = torch.matmul(torch.tensor(np.linalg.inv(K)), torch.stack((torch.tensor(x)*t_z2, torch.tensor(y)*t_z2, 1*t_z2), axis=0)[:,round_coord_cam2[1], round_coord_cam2[0]].reshape(3,-1))
            coord_world2 = (torch.tensor(np.linalg.inv(R)).float().matmul(coord_cam2) - torch.tensor(np.linalg.inv(R)).float().matmul(torch.tensor(T).float()))
            coord_world2_warp = torch.cat((coord_world2, torch.ones((1,valid_idx.shape[0]))), dim=0)
            coord_world2_trans = torch.matmul(trans3d, coord_world2_warp)
            coord_world2_trans = coord_world2_trans[:3] / coord_world2_trans[-1]
            loss = torch.mean((torch.tensor(pts_coord_world[:,valid_idx]).float() - coord_world2_trans)**2)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        with torch.no_grad():
            coord_cam2 = torch.matmul(torch.tensor(np.linalg.inv(K)), torch.stack((torch.tensor(x)*t_z2, torch.tensor(y)*t_z2, 1*t_z2), axis=0)[:,round_coord_cam2[1, border_valid_idx], round_coord_cam2[0, border_valid_idx]].reshape(3,-1))
            coord_world2 = (torch.tensor(np.linalg.inv(R)).float().matmul(coord_cam2) - torch.tensor(np.linalg.inv(R)).float().matmul(torch.tensor(T).float()))
            coord_world2_warp = torch.cat((coord_world2, torch.ones((1, border_valid_idx.shape[0]))), dim=0)
            coord_world2_trans = torch.matmul(trans3d, coord_world2_warp)
            coord_world2_trans = coord_world2_trans[:3] / coord_world2_trans[-1]
        trans3d = trans3d.detach().numpy()
        pts_coord_cam2 = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))[:,np.where(1-mask2.reshape(-1))[0]]
        camera_origin_coord_world2 = - np.linalg.inv(R).dot(T).astype(np.float32) 
        new_pts_coord_world2 = (np.linalg.inv(R).dot(pts_coord_cam2) - np.linalg.inv(R).dot(T)).astype(np.float32)
        new_pts_coord_world2_warp = np.concatenate((new_pts_coord_world2, np.ones((1, new_pts_coord_world2.shape[1]))), axis=0)
        new_pts_coord_world2 = np.matmul(trans3d, new_pts_coord_world2_warp)
        new_pts_coord_world2 = new_pts_coord_world2[:3] / new_pts_coord_world2[-1]
        new_pts_colors2 = (np.array(image_curr).reshape(-1,3).astype(np.float32)/255.)[np.where(1-mask2.reshape(-1))[0]]
    
        vector_camorigin_to_campixels = coord_world2_trans.detach().numpy() - camera_origin_coord_world2
        vector_camorigin_to_pcdpixels = pts_coord_world[:,valid_idx[border_valid_idx]] - camera_origin_coord_world2
        
        compensate_depth_coeff = np.sum(vector_camorigin_to_pcdpixels * vector_camorigin_to_campixels, axis=0) / np.sum(vector_camorigin_to_campixels * vector_camorigin_to_campixels, axis=0) # N_correspond
        compensate_pts_coord_world2_correspond = camera_origin_coord_world2 + vector_camorigin_to_campixels * compensate_depth_coeff.reshape(1,-1)
    
        compensate_coord_cam2_correspond = R.dot(compensate_pts_coord_world2_correspond) + T
        homography_coord_cam2_correspond = R.dot(coord_world2_trans.detach().numpy()) + T
        compensate_depth_correspond = compensate_coord_cam2_correspond[-1] - homography_coord_cam2_correspond[-1] # N_correspond
        compensate_depth_zero = np.zeros(4)
        compensate_depth = np.concatenate((compensate_depth_correspond, compensate_depth_zero), axis=0)  # N_correspond+4
        pixel_cam2_correspond = pixel_coord_cam2[:, border_valid_idx] # 2, N_correspond (xy)
        pixel_cam2_zero = np.array([[0,0,W-1,W-1],[0,H-1,0,H-1]])
        pixel_cam2 = np.concatenate((pixel_cam2_correspond, pixel_cam2_zero), axis=1).transpose(1,0) # N+H, 2
        masked_pixels_xy = np.stack(np.where(1-mask2), axis=1)[:, [1,0]]
        new_depth_linear, new_depth_nearest = interp_grid(pixel_cam2, compensate_depth, masked_pixels_xy), interp_grid(pixel_cam2, compensate_depth, masked_pixels_xy, method='nearest')
        new_depth = np.where(np.isnan(new_depth_linear), new_depth_nearest, new_depth_linear)
        pts_coord_cam2 = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))[:,np.where(1-mask2.reshape(-1))[0]]
        x_nonmask, y_nonmask = x.reshape(-1)[np.where(1-mask2.reshape(-1))[0]], y.reshape(-1)[np.where(1-mask2.reshape(-1))[0]]
        compensate_pts_coord_cam2 = np.matmul(np.linalg.inv(K), np.stack((x_nonmask*new_depth, y_nonmask*new_depth, 1*new_depth), axis=0))
        new_warp_pts_coord_cam2 = pts_coord_cam2 + compensate_pts_coord_cam2
        new_pts_coord_world2 = (np.linalg.inv(R).dot(new_warp_pts_coord_cam2) - np.linalg.inv(R).dot(T)).astype(np.float32)
        new_pts_coord_world2_warp = np.concatenate((new_pts_coord_world2, np.ones((1, new_pts_coord_world2.shape[1]))), axis=0)
        new_pts_coord_world2 = np.matmul(trans3d, new_pts_coord_world2_warp)
        new_pts_coord_world2 = new_pts_coord_world2[:3] / new_pts_coord_world2[-1]
        new_pts_colors2 = (np.array(image_curr).reshape(-1,3).astype(np.float32)/255.)[np.where(1-mask2.reshape(-1))[0]]
        pts_coord_world = np.concatenate((pts_coord_world, new_pts_coord_world2), axis=-1) ### Same with inv(c2w) * cam_coord (in homogeneous space)
        pts_colors = np.concatenate((pts_colors, new_pts_colors2), axis=0)



    yz_reverse = np.array([[1,0,0], [0,-1,0], [0,0,-1]])
    traindata = {
                'camera_angle_x': fov[0],
                'W': W,
                'H': H,
                'pcd_points': pts_coord_world,
                'pcd_colors': pts_colors,
                'frames': [],
                }
    internel_render_poses = get_pcdGenPoses('hemisphere', {'center_depth': center_depth, 'degree': 20})
    iterable_align = range(len(render_poses))



    inpaint_maskj = None
    inpaint_imagej = None
    image_gen = None
    inpaint_output = None
    inpaint_prompt = ""
    for i in iterable_align:
        text = ""
        inputs = processor(style_ref_images[i], text, return_tensors="pt").to("cuda")
        out = blip_model.generate(**inputs)
        inpaint_prompt = processor.decode(out[0], skip_special_tokens=True)
        print(inpaint_prompt)
        
        for j in range(len(internel_render_poses)):
            idx = i * len(internel_render_poses) + j
            print(f'{idx+1} / {len(render_poses)*len(internel_render_poses)}')
    
            ### Transform world to pixel
            Rw2i = render_poses[i,:3,:3]
            Tw2i = render_poses[i,:3,3:4]
            Ri2j = internel_render_poses[j,:3,:3]
            Ti2j = internel_render_poses[j,:3,3:4]
    
            Rw2j = np.matmul(Ri2j, Rw2i)
            Tw2j = np.matmul(Ri2j, Tw2i) + Ti2j
    
            # Transfrom cam2 to world + change sign of yz axis
            Rj2w = np.matmul(yz_reverse, Rw2j).T
            Tj2w = -np.matmul(Rj2w, np.matmul(yz_reverse, Tw2j))
            Pc2w = np.concatenate((Rj2w, Tj2w), axis=1)
            Pc2w = np.concatenate((Pc2w, np.array([[0,0,0,1]])), axis=0)
    
            pts_coord_camj = Rw2j.dot(pts_coord_world) + Tw2j
            pixel_coord_camj = np.matmul(K, pts_coord_camj)
    
            valid_idxj = np.where(np.logical_and.reduce((pixel_coord_camj[2]>0, 
                                                            pixel_coord_camj[0]/pixel_coord_camj[2]>=0, 
                                                            pixel_coord_camj[0]/pixel_coord_camj[2]<=W-1, 
                                                            pixel_coord_camj[1]/pixel_coord_camj[2]>=0, 
                                                            pixel_coord_camj[1]/pixel_coord_camj[2]<=H-1)))[0]
            
            if len(valid_idxj) == 0:
                    continue
            pts_depthsj = pixel_coord_camj[-1:, valid_idxj]
            pixel_coord_camj = pixel_coord_camj[:2, valid_idxj]/pixel_coord_camj[-1:, valid_idxj]
            round_coord_camj = np.round(pixel_coord_camj).astype(np.int32)
    
            x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy') # pixels
            grid = np.stack((x,y), axis=-1).reshape(-1,2)
            imagej = interp_grid(pixel_coord_camj.transpose(1,0), pts_colors[valid_idxj], grid, method='linear', fill_value=0).reshape(H,W,3)
            imagej = edgemask[...,None]*imagej + (1-edgemask[...,None])*np.pad(imagej[1:-1,1:-1], ((1,1),(1,1),(0,0)), mode='edge')
    
            depthj = interp_grid(pixel_coord_camj.transpose(1,0), pts_depthsj.T, grid, method='linear', fill_value=0).reshape(H,W)
            depthj = edgemask*depthj + (1-edgemask)*np.pad(depthj[1:-1,1:-1], ((1,1),(1,1)), mode='edge')
    
            maskj = np.zeros((H,W), dtype=np.float32)
            maskj[round_coord_camj[1], round_coord_camj[0]] = 1
            maskj = maximum_filter(maskj, size=(9,9), axes=(0,1))
            imagej = maskj[...,None]*imagej + (1-maskj[...,None])*(-1)
    
            maskj = minimum_filter((imagej.sum(-1)!=-3)*1, size=(11,11), axes=(0,1))
            imagej = maskj[...,None]*imagej + (1-maskj[...,None])*0
            imagej = Image.fromarray(np.round(imagej*255.).astype(np.uint8))
            maskj = Image.fromarray((255 - maskj * 255).astype(np.uint8))
    
            if j != 2 and calculate_white_ratio(maskj) > 0.0001:
            #if j != 2:
                print('inpainting...')
                maskj = expand_white_pixels (maskj)
                inpaint_maskj = maskj.convert("RGB")
                #inpaint_maskj.show()
                print('image input:')
                #imagej.show()
                blurred_mask = pipe.mask_processor.blur(imagej, blur_factor=50)
                style_ref_image = style_ref_images[i]
                style_ref_image.resize((512, 512))
                #style_ref_image.show()
    
                images = ip_model.generate(pil_image=style_ref_image,
                                prompt=inpaint_prompt,
                                negative_prompt= "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                                scale=2.0,
                                guidance_scale=8,
                                num_samples=1,
                                num_inference_steps=24, 
                                image=imagej,
                                mask_image=inpaint_maskj,
                                strength=0.99
                                )
    
                imagej = images[0]
                #imagej.show()
            
    
            traindata['frames'].append({
                    'image': imagej, 
                    'transform_matrix': Pc2w.tolist(),
                })

    save_data(traindata, './traindata')

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training Preparation script parameters")
    
    parser.add_argument('--image_360_path', type=str, default="")

    args = parser.parse_args()

    torch.cuda.set_device(torch.device("cuda:0"))

    # Pass the parsed argument to the infer function
    infer(args.image_360_path)
    
    # All done
    print("\nPreparation of training dataset complete.")


