import os
import warnings
from random import randint
warnings.filterwarnings(action='ignore')
import numpy as np
from PIL import Image
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
import json
import copy
import logging
import trimesh
import pyvista
import fast_simplification
import shutil
import runpod


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def remove_directory(dir_path):
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"Directory '{dir_path}' has been removed successfully.")
        except Exception as e:
            print(f'Failed to remove {dir_path}. Reason: {e}')
    else:
        print(f"Directory {dir_path} does not exist.")

def generate_seed_hemisphere(center_depth, degree=5):
    # change later to test
    degree = 5
    thlist = np.array([degree, 0, 0, 0, -degree])
    philist = np.array([0, -degree, 0, degree, 0])
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        # curr_pose = np.zeros((1, 3, 4))
        d = center_depth # central point of (hemi)sphere / you can change this value
        
        render_poses[i,:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_poses[i,:3,3:4] = np.array([d*np.sin(th/180*np.pi), 0, d-d*np.cos(th/180*np.pi)]).reshape(3,1) + np.array([0, d*np.sin(phi/180*np.pi), d-d*np.cos(phi/180*np.pi)]).reshape(3,1)# Transition vector
        # render_poses[i,:3,3:4] = np.zeros((3,1))

    return render_poses

def generate_seed_360(viewangle, n_views):
    N = n_views
    render_poses = np.zeros((N, 3, 4))
    for i in range(N):
        th = (viewangle/N)*i/180*np.pi
        render_poses[i,:3,:3] = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        render_poses[i,:3,3:4] = np.random.randn(3,1)*0.0 # Transition vector

    return render_poses


class CameraParams:
    def __init__(self, H: int = 512, W: int = 512):
        self.H = H
        self.W = W
        self.focal = (5.8269e+02, 5.8269e+02)
        self.fov = (2*np.arctan(self.W / (2*self.focal[0])), 2*np.arctan(self.H / (2*self.focal[1])))
        self.K = np.array([
            [self.focal[0], 0., self.W/2],
            [0., self.focal[1], self.H/2],
            [0.,            0.,       1.],
        ]).astype(np.float32)

class GSParams: 
    def __init__(self):
        self.sh_degree = 3
        self.images = "images"
        self.resolution = -1
        self.white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.use_depth = False

        self.iterations = 2990#3_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 2990#3_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002

        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

def get_pcdGenPoses(pcdgenpath, argdict={}):
    if pcdgenpath == 'rotate360':
        render_poses = generate_seed_360(360, 10)
    elif pcdgenpath == 'hemisphere':
        render_poses = generate_seed_hemisphere(argdict['center_depth'])
    else:
        raise("Invalid pcdgenpath")
    return render_poses

opt = GSParams()
cam = CameraParams()
save_dir='./'
root = 'outputs'

bg_color = [1, 1, 1] if opt.white_background else [0, 0, 0]

get_kernel = lambda p: torch.ones(1, 1, p * 2 + 1, p * 2 + 1).to('cuda')
t2np = lambda x: (x[0].permute(1, 2, 0).clamp_(0, 1) * 255.0).to(torch.uint8).detach().cpu().numpy()
np2t = lambda x: (torch.as_tensor(x).to(torch.float32).permute(2, 0, 1) / 255.0)[None, ...].to('cuda')
pad_mask = lambda x, padamount=1: t2np(
    F.conv2d(np2t(x[..., None]), get_kernel(padamount), padding=padamount))[..., 0].astype(bool)


def map_to_sphere(x, y, z, W, H, f, yaw_radian, pitch_radian):


    theta = np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))
    phi = np.arctan2(y, x)

    # Apply rotation transformations here
    theta_prime = np.arccos(np.sin(theta) * np.sin(phi) * np.sin(pitch_radian) +
                            np.cos(theta) * np.cos(pitch_radian))

    phi_prime = np.arctan2(np.sin(theta) * np.sin(phi) * np.cos(pitch_radian) -
                           np.cos(theta) * np.sin(pitch_radian),
                           np.sin(theta) * np.cos(phi))
    phi_prime += yaw_radian
    phi_prime = phi_prime % (2 * np.pi)

    return theta_prime.flatten(), phi_prime.flatten()

def interpolate_color(coords, img, method='bilinear'):
    order = {'nearest': 0, 'bilinear': 1, 'bicubic': 3}.get(method, 1)
    red = map_coordinates(img[:, :, 0], coords, order=order, mode='reflect')
    green = map_coordinates(img[:, :, 1], coords, order=order, mode='reflect')
    blue = map_coordinates(img[:, :, 2], coords, order=order, mode='reflect')
    return np.stack((red, green, blue), axis=-1)


def panorama_to_plane(panorama_image, FOV, output_size, yaw, pitch):
    #panorama = Image.open(panorama_path).convert('RGB')

    if not isinstance(panorama_image, Image.Image):
        raise ValueError("The 'panorama_image' must be a PIL Image object")
    panorama = panorama_image.convert('RGB')
    
    pano_width, pano_height = panorama.size
    pano_array = np.array(panorama)
    yaw_radian = np.radians(yaw)
    pitch_radian = np.radians(pitch)

    W, H = output_size
    f = (0.5 * W) / np.tan(np.radians(FOV) / 2)

    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

    x = u - W / 2
    y = H / 2 - v
    z = f

    theta, phi = map_to_sphere(x, y, z,W, H, f, yaw_radian, pitch_radian)

    U = phi * pano_width / (2 * np.pi)
    V = theta * pano_height / np.pi

    U, V = U.flatten(), V.flatten()
    coords = np.vstack((V, U))

    colors = interpolate_color(coords, pano_array)
    output_image = Image.fromarray(colors.reshape((H, W, 3)).astype('uint8'), 'RGB')

    return output_image


def save_data(data, base_path):
    data_copy = copy.deepcopy(data)
    
    if 'frames' in data_copy:
        for i, frame in enumerate(data_copy['frames']):
            img_path = f'{base_path}/image_{i}.jpg'
            frame['image'].save(img_path)
            frame['image'] = img_path 

    if 'pcd_points' in data_copy:
        np.save(f'{base_path}/pcd_points.npy', data_copy['pcd_points'])
        data_copy['pcd_points'] = f'{base_path}/pcd_points.npy'
    
    if 'pcd_colors' in data_copy:
        np.save(f'{base_path}/pcd_colors.npy', data_copy['pcd_colors'])
        data_copy['pcd_colors'] = f'{base_path}/pcd_colors.npy'

    with open(f'{base_path}/data.json', 'w') as fp:
        json.dump(data_copy, fp, indent=4)


def run_command(command):
        try:
            print(f"Executing command: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, check=True,encoding='utf-8')
            logger.info(f"Command succeeded: {' '.join(command)}")
            logger.info(f"Standard Output: {result.stdout}")
            logger.info(f"Standard Error: {result.stderr}")
            if result.returncode == 0:
                print("GS Script executed successfully.")
            else:
                print("Script execution failed.")
                print(f"Error message: {result.stderr}")
                return {"error": result.stderr}
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {' '.join(command)}")
            logger.error(f"Return Code: {e.returncode}")
            logger.error(f"Standard Output: {e.stdout}")
            logger.error(f"Standard Error: {e.stderr}")
            return {"error": e.stderr}


def generate_and_extract_full_sphere_poses(n_horizontal, n_vertical):
    render_poses = []
    yaws = []
    pitches = []

    # Compute angle increments
    yaw_increment = 360 / n_horizontal
    pitch_increment = 180 / (n_vertical - 1)

    # Loop through pitches from 0 to ±90 degrees
    pitch_angles = [0] + [-i * pitch_increment for i in range(1, n_vertical//2+1)] + [i * pitch_increment for i in range(1, n_vertical//2+1)]

    #temp
    pitch_angles = [0,-90,90]
    
    for pitch in pitch_angles:
        # Determine number of yaw positions: at poles (±90 degrees), only generate one position
        num_yaw_positions = 1 if abs(pitch) == 90 else n_horizontal

        for j in range(num_yaw_positions):
            yaw = j * yaw_increment if num_yaw_positions > 1 else 0  # At poles, yaw is irrelevant
            
            yaws.append(yaw)
            pitches.append(pitch)
            
            pitch_rad = np.radians(pitch)
            yaw_rad = np.radians(yaw)
            
            Ry = np.array([
                [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
                [0, 1, 0],
                [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
            ])
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
                [0, np.sin(pitch_rad), np.cos(pitch_rad)]
            ])
            R = Ry @ Rx
            t = np.zeros((3, 1))
            # Pose matrix
            pose = np.hstack((R, t))
            render_poses.append(pose)

    return yaws, pitches, np.array(render_poses)

def run_script_in_directory(directory, script, args):
    # 保存当前目录
    original_directory = os.getcwd()
    
    # 更改工作目录
    os.chdir(directory)
    
    # 构建完整的命令
    command = ['python', script] + args
    
    try:
        # 执行脚本并等待完成
        result = subprocess.run(command, capture_output=True, text=True)
        
        # 检查执行结果
        if result.returncode == 0:
            print("Script executed successfully.")
            # 你可以在这里添加更多的日志记录或其他操作
        else:
            print("Script execution failed.")
            print(f"Error message: {result.stderr}")
    except Exception as e:
        print(f"An error occurred while executing the script: {str(e)}")
    finally:
        # 返回原始目录
        os.chdir(original_directory)

n_horizontal = 10  # Number of horizontal divisions
n_vertical = 5    # Number of vertical divisions
yaws, pitches, render_poses = generate_and_extract_full_sphere_poses(n_horizontal, n_vertical)

pitches = np.array(pitches)+90
yaws = -np.array(yaws)

background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
d_model = torch.hub.load('./ZoeDepth', 'ZoeD_N', source='local', pretrained=True).to('cuda')
controlnet = None
lama = None


def upload_to_s3(file_name, bucket_name, aws_access_key_id, aws_secret_access_key, region_name, object_name=None, expiration=3600):
    """
    Uploads a file to an S3 bucket using hard-coded AWS credentials.
    
    :param file_name: Full path to the file to upload.
    :param bucket_name: Name of the bucket where the file will be stored.
    :param object_name: S3 object name. If not specified, file_name is used.
    """
    if object_name is None:
        object_name = file_name

    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name  
    )

    try:
        # 上传文件
        response = s3_client.upload_file(file_name, bucket_name, object_name)
        presigned_url = s3_client.generate_presigned_url('get_object',
                                                         Params={'Bucket': bucket_name,
                                                                 'Key': object_name},
                                                         ExpiresIn=expiration)
        return {'url': presigned_url}
    except FileNotFoundError:
        return {'error': "The file was not found"}
    except NoCredentialsError:
        return {'error': "Credentials not available"}
    except Exception as e:
        return {'error': f"An error occurred: {e}"}



def d(im):
    return d_model.infer_pil(im)
#H, W, K = cam.H, cam.W, cam.K
H, W, K = 512, 512, cam.K
#FOV = 120
FOV = 120
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



def inference(event) -> Union[str, dict]:
    input_data = event.get("input", {})
    image_url = input_data.get("image_url", "")
    aws_access_key_id = input_data.get("aws_access_key_id", "")
    aws_secret_access_key = input_data.get("aws_secret_access_key", "")
    region_name = input_data.get("region_name", "")
    bucket_name = input_data.get("bucket_name", "")

    response = requests.get(image_url)
    image_360 = None

    if response.status_code == 200:
        image_360 = Image.open(BytesIO(response.content))
        #image_360.show()

        width, height = image_360.size
        if height / width != 0.5:
            return {"error": "Not a panorama image"}
    else:
        return {"error": f"Failed to retrieve image. Status code: {response.status_code}"}


    directory = "train_res"
    # Remove the directory
    remove_directory(directory)
    
    
    image_curr = panorama_to_plane(image_360, FOV, (H, W), yaws[0], pitches[0])
    #image_curr.show()
    w_in, h_in = image_curr.size
    depth_curr = d(image_curr)
    center_depth = np.mean(depth_curr[h_in//2-10:h_in//2+10, w_in//2-10:w_in//2+10])

    x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy') # pixels
    edgeN = 2

    edgemask = np.ones((H-2*edgeN, W-2*edgeN))
    edgemask = np.pad(edgemask, ((edgeN,edgeN),(edgeN,edgeN)))
    '''
    R0 是 render_poses 中的第一个 3×3 子矩阵，表示第一个视点的旋转矩阵

    T0 是与 R0 对应的 3×1 子矩阵，表示第一个视点的平移向量。
    在您的函数 generate_seed_360 中，平移向量初始化为零
    （np.random.randn(3,1)*0.0），这意味着没有平移，只有旋转
    '''
    R0, T0 = render_poses[0,:3,:3], render_poses[0,:3,3:4]
    # p cam=K−1 ⋅ [x⋅depth,y⋅depth,depth]T
    # 得到相机为圆心的3d depth map坐标
    pts_coord_cam = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))
    #将从depthmap相机坐标系中得到的点转换到世界坐标系中
    new_pts_coord_world2 = (np.linalg.inv(R0).dot(pts_coord_cam) - np.linalg.inv(R0).dot(T0)).astype(np.float32) ## new_pts_coord_world2
    # 初始化归一化颜色数据（不重要）
    new_pts_colors2 = (np.array(image_curr).reshape(-1,3).astype(np.float32)/255.) ## new_pts_colors2
    pts_coord_world, pts_colors = new_pts_coord_world2.copy(), new_pts_colors2.copy()
    iterable_dream = range(1, len(render_poses))

    traindata = {
            'camera_angle_x': fov[0],
            'W': W,
            'H': H,
            'pcd_points': [],
            'pcd_colors': [],
            'frames': [],
        }
    
    for i in iterable_dream:
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
    internel_render_poses = get_pcdGenPoses('hemisphere', {'center_depth': center_depth})
    iterable_align = range(len(render_poses))
    iterable_align

    

    for i in iterable_align:
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

            traindata['frames'].append({
                    'image': imagej, 
                    'transform_matrix': Pc2w.tolist(),
                })


    save_data(traindata, './traindata')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


    command = ["python", "train.py", "-s", "traindata", "-m", "train_res", "-r", "2"]
    run_command(command)
    
    fpath = '/workspace/gaussian-opacity-field/train_res/point_cloud/iteration_2900/point_cloud.ply'
    
    target_directory = '/workspace/point-cloud-tools'
    script_name = 'convert.py'
    splat_path = fpath.split(".")[0] + ".splat"
    script_args = [
            fpath,
            splat_path,
            '--ply_input_format=inria'
        ]
    

    run_script_in_directory(target_directory, script_name, script_args)
    
    s3_object_name = "test.splat"

    command = ["python", "extract_mesh.py", "-m", "train_res", "--iteration", "2900"]
    run_command(command)
    
    ply_mesh_path = '/workspace/gaussian-opacity-field/train_res/test/ours_2900/fusion/mesh_binary_search_7.ply'
    
    obj_path = 'test.obj'
    
    mesh = pyvista.read(ply_mesh_path)
    m  = fast_simplification.simplify_mesh(mesh, target_reduction = 0.999)
    m.save('simplifytest.ply')
    s_mesh = trimesh.load('simplifytest.ply', force='mesh')
    s_mesh.export(obj_path)

    splat_result = upload_to_s3(splat_path, bucket_name, aws_access_key_id, aws_secret_access_key, region_name, "test.splat")
    obj_result = upload_to_s3(obj_path, bucket_name, aws_access_key_id, aws_secret_access_key, region_name, "test.obj")


    if 'url' in splat_result:
        return {"spalt_url": splat_result['url'], "obj_url": obj_result['url']}
    else:
        return {"error": "Failed to upload:" + splat_result['error']}



runpod.serverless.start({"handler": inference})

'''
def main():

    event = {
        "input": {
            "image_url": "",
            "aws_access_key_id": "",
            "aws_secret_access_key": "",
            "region_name": "us-east-1",
            "bucket_name": ""

        }
    }

    result = inference(event)
    if "error" in result:
        print("Error:", result["error"])
    elif "spalt_url" in result:
        print(print("spalt_url:", result["spalt_url"]))
        print(print("obj_url:", result["obj_url"]))
    else:
        print(result)

if __name__ == "__main__":
    main()

'''