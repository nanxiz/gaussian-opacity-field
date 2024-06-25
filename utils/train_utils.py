import numpy as np
from scipy.spatial.transform import Rotation as RR
from PIL import Image, ImageDraw,ImageFilter
from scipy.ndimage import map_coordinates

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

def generate_seed_hemisphere(center_depth, degree=5):
    degree = 5
    thlist = np.array([degree, 0, 0, 0, -degree])
    philist = np.array([0, -degree, 0, degree, 0])
    psilist = np.array([0, 0, degree, -degree, 0])
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        psi = psilist[i]
        d = center_depth
        
        render_poses[i,:3,:3] = np.matmul(
            np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], 
                      [0, 1, 0], 
                      [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), 
            np.array([[1, 0, 0], 
                      [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], 
                      [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_poses[i,:3,3:4] = 4 * np.array([d*np.sin(th/180*np.pi), 
                                               d*np.sin(psi/180*np.pi), 
                                               d-d*np.cos(th/180*np.pi)]).reshape(3,1) + \
                                 np.array([0, d*np.sin(phi/180*np.pi), d-d*np.cos(phi/180*np.pi)]).reshape(3,1)
    return render_poses

def generate_seed_360(viewangle, n_views):
    N = n_views
    render_poses = np.zeros((N, 3, 4))
    for i in range(N):
        th = (viewangle/N)*i/180*np.pi
        render_poses[i,:3,:3] = np.array([[np.cos(th), 0, np.sin(th)], 
                                          [0, 1, 0], 
                                          [-np.sin(th), 0, np.cos(th)]])
        render_poses[i,:3,3:4] = np.random.randn(3,1)*0.0
    return render_poses

def get_pcdGenPoses(pcdgenpath, argdict={}):
    if pcdgenpath == 'rotate360':
        return generate_seed_360(360, 10)
    elif pcdgenpath == 'hemisphere':
        return generate_seed_hemisphere(argdict.get('center_depth', 10))
    else:
        raise ValueError("Invalid pcdgenpath")

def print_euler_angles_from_rotation_matrix(rotation_matrix):

    rot = RR.from_matrix(rotation_matrix)
    
    euler_angles = rot.as_euler('xyz', degrees=True) 
    
    print("Euler angles (degrees):")
    print("Rotation around x-axis (pitch): {:.2f} degrees".format(euler_angles[0]))
    print("Rotation around y-axis (yaw): {:.2f} degrees".format(euler_angles[1]))
    print("Rotation around z-axis (roll): {:.2f} degrees".format(euler_angles[2]))




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


def visualize_depth(depth):
    depth_normalized = 255 * (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth_normalized = depth_normalized.astype(np.uint8)

    depth_image = Image.fromarray(depth_normalized, mode='L')
    depth_image.show()

    return depth_image


def prepare_top_and_bottom_mask(diameter_ratio = 3, width=1024, height=1024):

    image = Image.new("L", (width, height), "black")
    draw = ImageDraw.Draw(image)

    diameter = width / diameter_ratio

    center_x, center_y = width // 2, height // 2

    left = center_x - diameter // 2
    top = center_y - diameter // 2
    right = center_x + diameter // 2
    bottom = center_y + diameter // 2

    draw.ellipse([left, top, right, bottom], fill="white")

    return image.convert("RGB")


def calculate_white_ratio(image):
    """
    Calculate the ratio of white pixels to the total number of pixels in a black and white image.

    Args:
        image (PIL.Image): A black and white image (mode "1" or "L").

    Returns:
        float: The ratio of white pixels in the image.
    """
    if image.mode == 'RGB':
        image = image.convert('L')
        
    if image.mode != 'L':
        raise ValueError("Image must be in 'L' mode for black and white images after conversion.")

    pixel_data = image.getdata()

    white_count = sum(1 for pixel in pixel_data if pixel == 255)

    total_pixels = image.width * image.height
    print(total_pixels)

    white_ratio = white_count / total_pixels
    print(white_count)

    return white_ratio




def expand_white_pixels(image, radius=4):
    """
    Expand the white pixels in a binary (black and white) PIL image.

    Args:
        image (PIL.Image): A black and white image (mode '1' or 'L').

    Returns:
        PIL.Image: A new image with expanded white pixels.
    """
    if image.mode not in ['1', 'L']:
        raise ValueError("Image must be in '1' or 'L' mode for black and white images.")
    white_expand = image.filter(ImageFilter.MaxFilter(size=2*radius + 1))
    return white_expand





