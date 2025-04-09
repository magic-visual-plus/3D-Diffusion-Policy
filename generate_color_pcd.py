import numpy as np
import cv2
import open3d as o3d
import visualizer

def grid_sample_pcd(point_cloud, grid_size=0.005):
    """
    A simple grid sampling function for point clouds.

    Parameters:
    - point_cloud: A NumPy array of shape (N, 3) or (N, 6), where N is the number of points.
                   The first 3 columns represent the coordinates (x, y, z).
                   The next 3 columns (if present) can represent additional attributes like color or normals.
    - grid_size: Size of the grid for sampling.

    Returns:
    - A NumPy array of sampled points with the same shape as the input but with fewer rows.
    """
    coords = point_cloud[:, :3]  # Extract coordinates
    scaled_coords = coords / grid_size
    grid_coords = np.floor(scaled_coords).astype(int)
    
    # Create unique grid keys
    keys = grid_coords[:, 0] + grid_coords[:, 1] * 10000 + grid_coords[:, 2] * 100000000
    
    # Select unique points based on grid keys
    _, indices = np.unique(keys, return_index=True)
    
    # Return sampled points
    return point_cloud[indices]

def create_colored_point_cloud(color, depth, far=1.0, near=0.1, num_points=10000):
        scale = 1
        fx=730.9765625  # fx
        fy=731.890625   # fy
        cx=497.4765625  # cx
        cy=402.7109375  # cy
        assert(depth.shape[0] == color.shape[0] and depth.shape[1] == color.shape[1])
    
        # Create meshgrid for pixel coordinates
        xmap = np.arange(color.shape[1])
        ymap = np.arange(color.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)

        # Calculate 3D coordinates
        points_z = depth / scale
        points_x = (xmap - cx) * points_z / fx
        points_y = (ymap - cy) * points_z / fy
        cloud = np.stack([points_x, points_y, points_z], axis=-1)
        cloud = cloud.reshape([-1, 3])
        
        # Clip points based on depth
        # mask = (cloud[:, 2] < far) & (cloud[:, 2] > near)
        # cloud = cloud[mask]
        color = color.reshape([-1, 3])
        # color = color[mask]


        colored_cloud = np.hstack([cloud, color.astype(np.float32)])
        if False:
            colored_cloud = grid_sample_pcd(colored_cloud, grid_size=0.005)
        
        if num_points > colored_cloud.shape[0]:
            num_pad = num_points - colored_cloud.shape[0]
            pad_points = np.zeros((num_pad, 6))
            colored_cloud = np.concatenate([colored_cloud, pad_points], axis=0)
        else: 
            # Randomly sample points
            selected_idx = np.random.choice(colored_cloud.shape[0], num_points, replace=True)
            colored_cloud = colored_cloud[selected_idx]
        
        # shuffle
        np.random.shuffle(colored_cloud)
        return colored_cloud

# 读取image和npy数据
color_image = cv2.imread("color_image.png")
depth_image = np.load("depth_image.npy")

colored_point_cloud = create_colored_point_cloud(color_image, depth_image, num_points=10000)

print(colored_point_cloud.shape)  # 输出点云的形状
visualizer.visualize_pointcloud(colored_point_cloud)