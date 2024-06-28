import os
import warnings
from random import randint
warnings.filterwarnings(action='ignore')
import numpy as np
from PIL import Image
#from tqdm import tqdm
from PIL import Image
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
import logging
import trimesh
import pyvista
import fast_simplification
import shutil
import runpod


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
torch.cuda.empty_cache()

def remove_directory(dir_path):
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"Directory '{dir_path}' has been removed successfully.")
        except Exception as e:
            print(f'Failed to remove {dir_path}. Reason: {e}')
    else:
        print(f"Directory {dir_path} does not exist.")


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




image_360_path = "zcabnzh__00000_.png"
bucket_name = 'splatminiworlds'



def inference(event) -> Union[str, dict]:
    torch.cuda.empty_cache()
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

    

    image_360_path = 'image_360.png'

    image_360.save(image_360_path)

    command = ["python", "prepare_training.py", "--image_360_path", image_360_path]
    run_command(command)





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
    
    splat_result = upload_to_s3(splat_path, bucket_name, aws_access_key_id, aws_secret_access_key, region_name, "test.splat")

    command = ["python", "extract_mesh.py", "-m", "train_res", "--iteration", "2900"]
    run_command(command)
    
    ply_mesh_path = '/workspace/gaussian-opacity-field/train_res/test/ours_2900/fusion/mesh_binary_search_7.ply'
    
    obj_path = 'test.obj'
    
    mesh = pyvista.read(ply_mesh_path)
    m  = fast_simplification.simplify_mesh(mesh, target_reduction = 0.999)
    m.save('simplifytest.ply')
    s_mesh = trimesh.load('simplifytest.ply', force='mesh')
    s_mesh.export(obj_path)

    #splat_result = upload_to_s3(splat_path, bucket_name, aws_access_key_id, aws_secret_access_key, region_name, "test.splat")
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
            "bucket_name": "splatminiworlds"

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