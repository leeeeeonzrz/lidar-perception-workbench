#!/usr/bin/env python
import random
import rospy
import time
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import os


import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN



# 配置文件路径
SAVE_PATH = './output/pointcloud/'


def save_point_cloud_as_pcd(open3d_cloud, filename):
    # 将Open3D点云对象保存为PCD文件
    try:
        # 确保保存路径存在
        folder= SAVE_PATH+time.strftime("%Y%m%d_%H", time.localtime())+'/'
        
        os.makedirs(folder, exist_ok=True)
        full_path = os.path.join( folder, filename)
        o3d.io.write_point_cloud(full_path, open3d_cloud)
        rospy.loginfo(f"Point cloud saved to: {full_path}")
    except Exception as e:
        rospy.logerr(f"Error saving point cloud to {filename}: {e}")




def execute_global_registration(source, target, voxel_size):
    """
    执行全局配准（使用ICP算法）
    """
    # 估计法线
    # 使用混合搜索参数的KD树进行法线估计，这里指定了搜索的半径和最大近邻数
    # 参数radius设为体素大小的两倍，max_nn设为30，这样的设置旨在平衡计算效率和准确性
    # 估算源点云的法线
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    # 估算目标点云的法线
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    # 设置ICP配准参数
    icp = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000) # 设置最大迭代次数为2000次
    # 执行配准
    initial_guess = np.identity(4)
    result = o3d.pipelines.registration.registration_icp(
        source, target, voxel_size, initial_guess,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        icp)
    return result

def rotate_point_cloud_around_x_axis(point_cloud, angle_degrees):
    """
    将点云绕X轴旋转指定角度
    :param point_cloud: 输入的点云
    :param angle_degrees: 旋转角度（度）
    :return: 旋转后的点云
    """
    # 将角度转换为弧度
    angle_radians = np.radians(angle_degrees)
    
    # 创建绕X轴旋转的变换矩阵
    rotation_matrix = np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle_radians), -np.sin(angle_radians), 0],
        [0, np.sin(angle_radians), np.cos(angle_radians), 0],
        [0, 0, 0, 1]
    ])
    
    # 应用变换矩阵
    rotated_point_cloud = point_cloud.transform(rotation_matrix)
    return rotated_point_cloud

def rotate_point_cloud_around_y_axis(point_cloud, angle_degrees):
    """
    将点云绕Y轴旋转指定角度
    :param point_cloud: 输入的点云
    :param angle_degrees: 旋转角度（度）
    :return: 旋转后的点云
    """
    # 将角度转换为弧度
    angle_radians = np.radians(angle_degrees)
    
    # 创建绕Y轴旋转的变换矩阵
    rotation_matrix = np.array([
        [np.cos(angle_radians), 0, np.sin(angle_radians), 0],
        [0, 1, 0, 0],
        [-np.sin(angle_radians), 0, np.cos(angle_radians), 0],
        [0, 0, 0, 1]
    ])
    
    # 应用变换矩阵
    rotated_point_cloud = point_cloud.transform(rotation_matrix)
    return rotated_point_cloud

def plot_z_histogram(point_cloud, method='sturges'):
    """
    绘制点云在z轴方向上的点云数量直方图
    :param point_cloud: 输入的点云
    :param method: 选择自适应bin数量的方法，可选值为'sturges', 'freedman-diaconis', 'scott'
    Sturges' Rule：适用于小样本数据集。
    Freedman-Diaconis Rule：适用于中等样本数据集。
    Scott's Rule：适用于正态分布的数据集
    :return: z_values, bins, counts
    """
    points = np.asarray(point_cloud.points)
    z_values = points[:, 2]
    
    if method == 'sturges':
        bins = int(1 + 3.322 * np.log10(len(z_values)))
    elif method == 'freedman-diaconis':
        iqr = np.percentile(z_values, 75) - np.percentile(z_values, 25)
        h = 2 * iqr / (len(z_values) ** (1/3))
        bins = int((np.max(z_values) - np.min(z_values)) / h)
    elif method == 'scott':
        sigma = np.std(z_values)
        h = 3.5 * sigma / (len(z_values) ** (1/3))
        bins = int((np.max(z_values) - np.min(z_values)) / h)
    else:
        raise ValueError("Method must be 'sturges', 'freedman-diaconis', or 'scott'")
    
    counts, bin_edges = np.histogram(z_values, bins=bins)
    
    plt.figure(figsize=(10, 6))
    plt.hist(z_values, bins=bins, color='blue', alpha=0.7)
    plt.title(f'Z Axis Point Cloud Distribution (Method: {method})')
    plt.xlabel('Z Coordinate')
    plt.ylabel('Number of Points')
    plt.grid(True)
    plt.show()
    
    return z_values, bin_edges, counts

def determine_layers_from_histogram(counts, bin_edges, num_layers=3):
    """
    根据直方图确定层数
    :param counts: 直方图的计数
    :param bin_edges: 直方图的bin边缘
    :param num_layers: 目标层数
    :return: z_intervals
    """
    # 找到直方图的峰值
    peaks, _ = find_peaks(counts)
    
    # 如果峰值数量少于目标层数，使用等间距划分
    if len(peaks) < num_layers:
        z_intervals = np.linspace(bin_edges[0], bin_edges[-1], num_layers + 1)
    else:
        # 选择前num_layers个最大的峰值
        peak_indices = np.argsort(counts[peaks])[-(num_layers-1):]
        peak_positions = bin_edges[peaks[peak_indices]]
        
        # 插入直方图的最小值和最大值
        peak_positions = np.insert(peak_positions, 0, bin_edges[0])
        peak_positions = np.append(peak_positions, bin_edges[-1])
        
        # 对峰值位置进行排序
        z_intervals = np.sort(peak_positions)
    
    return z_intervals

def determine_layers_from_histogram_min(counts, bin_edges, num_layers=3):
    """
    根据直方图确定层数
    :param counts: 直方图的计数
    :param bin_edges: 直方图的bin边缘
    :param num_layers: 目标层数
    :return: z_intervals
    """
    # 找到直方图的极小值
    valleys, _ = find_peaks(-counts)
    
    # 如果极小值数量少于目标层数减一，使用等间距划分
    if len(valleys) < num_layers - 1:
        z_intervals = np.linspace(bin_edges[0], bin_edges[-1], num_layers + 1)
    else:
        # 选择前num_layers-1个最小的极小值
        valley_indices = np.argsort(counts[valleys])[:num_layers-1]
        valley_positions = bin_edges[valleys[valley_indices]]
        
        # 插入直方图的最小值和最大值
        valley_positions = np.insert(valley_positions, 0, bin_edges[0])
        valley_positions = np.append(valley_positions, bin_edges[-1])
        
        # 对极小值位置进行排序
        z_intervals = np.sort(valley_positions)
    
    return z_intervals
# def colorize_point_cloud_by_z_intervals(point_cloud, z_intervals):
#     """
#     根据点云的z坐标在z轴方向上划分层次，并为不同的高度赋予不同的颜色
#     :param point_cloud: 输入的点云
#     :param z_intervals: z轴方向的层次区间
#     :return: 赋色后的点云
#     """
#     points = np.asarray(point_cloud.points)
#     colors = np.zeros_like(points)
    
#     # 创建一个色谱
#     cmap = plt.get_cmap('viridis')  # 可以选择其他色谱，如 'plasma', 'inferno', 'magma', 'cividis'

#     # 为每个层次分配颜色
#     for i in range(len(z_intervals) - 1):
#         mask = [False] * len(points)
#         mask = (points[:, 2] >= z_intervals[i]) & (points[:, 2] < z_intervals[i + 1])
#         # colors[mask] = np.random.rand(3)  # 随机颜色
#         # 使用色谱分配颜色
#         colors[mask] = cmap(i / (len(z_intervals) - 2))[:3]  # [:3] 是为了获取 RGB 值
    
#     point_cloud.colors = o3d.utility.Vector3dVector(colors)
#     return point_cloud

def colorize_point_cloud_by_z_intervals(point_cloud, z_intervals):
    """
    根据点云的z坐标在z轴方向上划分层次，并为不同的高度赋予不同的颜色
    :param point_cloud: 输入的点云
    :param z_intervals: z轴方向的层次区间
    :return: 赋色后的点云
    """
    if not isinstance(point_cloud, o3d.geometry.PointCloud):
        raise ValueError("point_cloud 必须是 open3d.geometry.PointCloud 类型")
    
    if not (isinstance(z_intervals, list) or isinstance(z_intervals, np.ndarray)):
        raise ValueError("z_intervals 必须是列表或 numpy 数组")
    
    if len(z_intervals) < 2:
        raise ValueError("z_intervals 至少需要两个元素")
    
    if not all(earlier <= later for earlier, later in zip(z_intervals, z_intervals[1:])):
        raise ValueError("z_intervals 必须是严格递增的")
    
    points = np.asarray(point_cloud.points)
    if points.size == 0:
        return point_cloud
    
    colors = np.zeros_like(points)
    
    # 创建一个色谱
    cmap = plt.get_cmap('viridis')  # 可以选择其他色谱，如 'plasma', 'inferno', 'magma', 'cividis'
    
    # 对 z 值进行四舍五入处理，减少浮点数精度问题
    rounded_points_z = np.round(points[:, 2], decimals=5)
    
    # 为每个层次分配颜色
    for i in range(len(z_intervals) - 1):
        mask = (rounded_points_z >= z_intervals[i]) & (rounded_points_z < z_intervals[i + 1])
        colors[mask] = cmap(i / (len(z_intervals) - 2))[:3]  # [:3] 是为了获取 RGB 值
    
    # 处理最后一个区间的边界条件
    last_interval_mask = rounded_points_z >= z_intervals[-1]
    colors[last_interval_mask] = cmap((len(z_intervals) - 2) / (len(z_intervals) - 2))[:3]
    
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def plot_point_cloud_with_z_intervals(point_cloud, z_intervals):
    """
    绘制点云并添加指定的 z 区间平面
    :param point_cloud: 输入的点云
    :param z_intervals: z轴方向的层次区间
    """
    if not isinstance(point_cloud, o3d.geometry.PointCloud):
        raise ValueError("point_cloud 必须是 open3d.geometry.PointCloud 类型")
    
    if not (isinstance(z_intervals, list) or isinstance(z_intervals, np.ndarray)):
        raise ValueError("z_intervals 必须是列表或 numpy 数组")
    
    if len(z_intervals) < 2:
        raise ValueError("z_intervals 至少需要两个元素")
    
    if not all(earlier <= later for earlier, later in zip(z_intervals, z_intervals[1:])):
        raise ValueError("z_intervals 必须是严格递增的")
    
    points = np.asarray(point_cloud.points)
    if points.size == 0:
        return
    
    # 创建一个色谱
    cmap = plt.get_cmap('viridis')  # 可以选择其他色谱，如 'plasma', 'inferno', 'magma', 'cividis'
    
    # 为每个层次分配颜色
    colors = np.zeros_like(points)
    for i in range(len(z_intervals) - 1):
        mask = (points[:, 2] >= z_intervals[i]) & (points[:, 2] < z_intervals[i + 1])
        colors[mask] = cmap(i / (len(z_intervals) - 2))[:3]  # [:3] 是为了获取 RGB 值
    
    # 处理最后一个区间的边界条件
    last_interval_mask = points[:, 2] >= z_intervals[-1]
    colors[last_interval_mask] = cmap((len(z_intervals) - 2) / (len(z_intervals) - 2))[:3]
    
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    # 绘制点云
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
    
    # 绘制 z_intervals 平面
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    for z in z_intervals:
        xx, yy = np.meshgrid([x_min, x_max], [y_min, y_max])
        zz = np.full_like(xx, z)
        ax.plot_surface(xx, yy, zz, color='gray', alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Point Cloud with Z Intervals')
    plt.show()


def extract_ground_RANSAC(point_cloud, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """
    使用 RANSAC 算法从点云中提取地面
    :param point_cloud: 输入的点云
    :param distance_threshold: 点到平面的距离阈值
    :param ransac_n: 每次迭代中随机选择的点数
    :param num_iterations: RANSAC 迭代次数
    :return: 地面点云, 非地面点云
    """
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=distance_threshold,
                                                     ransac_n=ransac_n,
                                                     num_iterations=num_iterations)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    inlier_cloud = point_cloud.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])  # 红色表示地面

    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([0, 1.0, 0])  # 绿色表示非地面

    return inlier_cloud, outlier_cloud



def extract_ground_RANSAC_NormalVectorConstraint(point_clouds, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """
    使用 RANSAC 算法从点云中提取地面，并旋转点云使地面法向量接近（0, 0, 1）
    :param point_cloud: 输入的点云
    :param distance_threshold: 点到平面的距离阈值
    :param ransac_n: 每次迭代中随机选择的点数
    :param num_iterations: RANSAC 迭代次数
    :return: 地面点云, 非地面点云
    """
    # 创建一个新的点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_clouds)
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=distance_threshold,
                                                     ransac_n=ransac_n,
                                                     num_iterations=num_iterations)
    [a, b, c, d] = plane_model
    print(f"Original Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # 当前地面法向量
    current_normal = np.array([a, b, c])
    # 目标法向量
    target_normal = np.array([0, 0, 1])

    # # 计算旋转矩阵
    # v = np.cross(current_normal, target_normal) #1.计算两个法向量的叉积 v，得到旋转轴。
    # s = np.linalg.norm(v) #2.计算叉积的模长 s，表示旋转角度的正弦值。
    # c = np.dot(current_normal, target_normal) #3.计算两个法向量的点积 c，表示旋转角度的余弦值。
    # skew_symmetric_cross_v = np.array([[0, -v[2], v[1]],
    #                                    [v[2], 0, -v[0]],
    #                                    [-v[1], v[0], 0]]) #4. 构建反对称矩阵 skew_symmetric_cross_v。
    # #5.使用 Rodrigues 旋转公式计算旋转矩阵。  R=I+sinθK+(1−cosθ)K^2  ，K 是反对称矩阵，由两个法向量的叉积 ( v ) 构建
    # rotation_matrix = np.eye(3) + skew_symmetric_cross_v + np.dot(skew_symmetric_cross_v, skew_symmetric_cross_v) * ((1 - c) / (s ** 2))

    # 计算旋转轴 : 旋转轴为两个法向量的叉积
    axis = np.cross(current_normal, target_normal)
    axis = axis / np.linalg.norm(axis)  # 归一化旋转轴

    # 计算旋转角
    theta = np.arccos(np.dot(current_normal, target_normal) / (np.linalg.norm(current_normal) * np.linalg.norm(target_normal)))

    # Rodrigues旋转公式
    def rodrigues_rotation(axis, theta):
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        # 使用 Rodrigues 旋转公式计算旋转矩阵。  R=I+sinθK+(1−cosθ)K^2  ，K 是反对称矩阵，由两个法向量的叉积 ( v ) 构建
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
        return R

    # 计算旋转矩阵
    rotation_matrix = rodrigues_rotation(axis, theta)

    print("旋转矩阵:\n", rotation_matrix)


    # 应用旋转矩阵到点云
    point_cloud.rotate(rotation_matrix, center=point_cloud.get_center())

    # 重新计算地面平面方程
    rotated_inliers = point_cloud.select_by_index(inliers)
    rotated_plane_model, _ = rotated_inliers.segment_plane(distance_threshold=distance_threshold,
                                                           ransac_n=ransac_n,
                                                           num_iterations=num_iterations)
    [a_rot, b_rot, c_rot, d_rot] = rotated_plane_model
    print(f"Rotated Plane equation: {a_rot:.2f}x + {b_rot:.2f}y + {c_rot:.2f}z + {d_rot:.2f} = 0")

    inlier_cloud = point_cloud.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])  # 红色表示地面

    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([0, 1.0, 0])  # 绿色表示非地面

    return inlier_cloud, outlier_cloud, inliers

def extract_ground_height(point_cloud, height_threshold=0.5):
    """
    根据高度阈值提取地面点云。

    参数:
    point_cloud: numpy数组，表示输入的点云数据。
    height_threshold: 浮点数，表示高度阈值，默认为0.5。

    返回:
    ground_points: numpy数组，表示提取出的地面点云。
    non_ground_points: numpy数组，表示非地面点云。
    """
    # 通过高度阈值分割地面点和非地面点
    ground_points = point_cloud[point_cloud[:, 2] < height_threshold]
    non_ground_points = point_cloud[point_cloud[:, 2] >= height_threshold]

    ground_points.paint_uniform_color([1.0, 0, 0])  # 红色表示地面
    non_ground_points.paint_uniform_color([0, 1.0, 0])  # 绿色表示非地面
    return ground_points, non_ground_points

def extract_ground_normal(point_cloud, normal_threshold=0.001):
    """
    根据法线估计提取地面点云。
    原理：计算每个点的法线方向，假设地面点的法线接近垂直于重力方向（即法线向量接近 [0, 0, 1] 或 [0, 0, -1]）。
    优点：对倾斜地面有一定的适应性。
    缺点：需要计算法线，计算成本较高。
    
    参数:
    point_cloud: open3d.geometry.PointCloud 对象，表示输入的点云数据。
    normal_threshold: 浮点数，表示法线阈值，默认为0.85。

    返回:
    ground_points: numpy数组，表示提取出的地面点云。
    non_ground_points: numpy数组，表示非地面点云。
    """
    # 估计点云的法线
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=10))
    
    # 获取法线数据
    normals = np.asarray(point_cloud.normals)
    
    # 使用法线阈值创建地面掩码
    ground_mask = np.abs(normals[:, 2]) > normal_threshold
    
    # 分割地面点和非地面点
    ground_points = point_cloud.select_by_index(np.where(ground_mask)[0])
    non_ground_points = point_cloud.select_by_index(np.where(~ground_mask)[0])

    ground_points.paint_uniform_color([1.0, 0, 0])  # 红色表示地面
    non_ground_points.paint_uniform_color([0, 1.0, 0])  # 绿色表示非地面
    return ground_points, non_ground_points


def quaternion_to_transformation_matrix(quaternion):
    """
    将四元数转换为4x4的变换矩阵。
    
    :param quaternion: 四元数，格式为[w, x, y, z]
    :return: 4x4的变换矩阵
    """
    w, x, y, z = quaternion
    rotation_matrix = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y), 0],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x), 0],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2), 0],
        [0, 0, 0, 1]
    ])
    return rotation_matrix

def rotate_points_around_x_axis(points, angle_degrees):
    """
    旋转点集绕x轴旋转指定角度。
    
    参数:
    points: numpy数组，表示要旋转的点集。
    angle_degrees: 浮点数，表示旋转的角度，单位为度。
    
    返回:
    rotated_points: numpy数组，表示旋转后的点集。
    """
    
    # 将角度从度转换为弧度，以便使用三角函数。
    angle_radians = np.radians(angle_degrees)
    
    # 定义绕x轴旋转的旋转矩阵。
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle_radians), -np.sin(angle_radians)],
        [0, np.sin(angle_radians), np.cos(angle_radians)]
    ])
    
    # 通过点乘原始点集和旋转矩阵的转置来计算旋转后的点集。
    rotated_points = np.dot(points, rotation_matrix.T)
    
    # 返回旋转后的点集。
    return rotated_points


def rotate_points_around_y_axis(points, angle_degrees):
    """
    将点云绕Y轴旋转指定角度
    :param points: 输入的点云
    :param angle_degrees: 旋转角度（度）
    :return: 旋转后的点云
    """
    # 将角度转换为弧度
    angle_radians = np.radians(angle_degrees)
    
    # 创建绕Y轴旋转的变换矩阵
    rotation_matrix = np.array([
        [np.cos(angle_radians), 0, np.sin(angle_radians)],
        [0, 1, 0]
        [-np.sin(angle_radians), 0, np.cos(angle_radians)],
    ])
    
    # 应用变换矩阵
    # 通过点乘原始点集和旋转矩阵的转置来计算旋转后的点集。
    rotated_points = np.dot(points, rotation_matrix.T)
    
    # 返回旋转后的点集。
    return rotated_points


def rotate_points_around_z_axis(points, angle_degrees):
    """
    将点云绕Z轴旋转指定角度
    :param points: 输入的点云
    :param angle_degrees: 旋转角度（度）
    :return: 旋转后的点云
    """
    # 将角度转换为弧度
    angle_radians = np.radians(angle_degrees)
    
    # 创建绕Z轴旋转的变换矩阵
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians), np.cos(angle_radians), 0],
        [0, 0, 1]
    ])

    # 通过点乘原始点集和旋转矩阵的转置来计算旋转后的点集。
    rotated_points = np.dot(points, rotation_matrix.T)
    
    # 返回旋转后的点集。
    return rotated_points

def fit_plane_ransac(points, threshold=0.05, max_iterations=1000, min_inliers_ratio=0.1):
    """
    使用 RANSAC 算法从 3D 点云中拟合一个平面。

    Args:
        points (np.ndarray): 输入点云，形状为 (N, 3) 的 NumPy 数组。
        threshold (float): 点到平面的最大距离阈值，小于此阈值的点被视为内点 (inlier)。
        max_iterations (int): RANSAC 算法的最大迭代次数。
        min_inliers_ratio (float): 有效模型所需的最小内点比例。

    Returns:
        tuple: (best_plane, best_inlier_indices)
            - best_plane (tuple or None): 最佳拟合平面的参数 (a, b, c, d)，满足 ax+by+cz+d=0。
                                           法向量 (a, b, c) 已被归一化 (sqrt(a^2+b^2+c^2)=1)。
                                           如果找不到合适的平面，则返回 None。
            - best_inlier_indices (np.ndarray or None): 对应于最佳平面的内点的索引数组。
                                                        如果找不到合适的平面，则返回 None。
    """
    num_points = points.shape[0]
    if num_points < 3:
        print("错误：至少需要3个点才能拟合平面。")
        return None, None

    best_plane = None          # 存储最佳平面参数 (a, b, c, d)
    best_inlier_indices = None # 存储最佳内点索引
    max_inliers = -1           # 存储找到的最大内点数量

    min_inliers_count = int(num_points * min_inliers_ratio) # 模型有效所需的最少内点数

    print(f"RANSAC 开始：最大迭代次数={max_iterations}，距离阈值={threshold}，最少内点数={min_inliers_count}")

    for i in range(max_iterations):
        # 1. 随机采样 3 个点
        try:
            sample_indices = random.sample(range(num_points), 3)
            sample_points = points[sample_indices]
        except ValueError:
            # 如果总点数少于3，理论上不会发生，但作为安全检查
            print("警告：没有足够的唯一 K 点进行采样。")
            continue

        # 2. 根据采样点拟合一个平面模型
        p1, p2, p3 = sample_points
        v1 = p2 - p1  # 平面上的向量1
        v2 = p3 - p1  # 平面上的向量2
        # 计算法向量：两个平面向量的叉积
        normal = np.cross(v1, v2)
        normal_norm = np.linalg.norm(normal) # 法向量的模长

        # 检查采样点是否共线 (法向量长度接近0)
        if normal_norm < 1e-6:
            # print(f"迭代 {i+1}: 采样点共线，跳过。")
            continue

        # 归一化法向量
        normal /= normal_norm

        # 计算平面参数 d: ax + by + cz + d = 0  => d = - (ax + by + cz)
        # 使用第一个采样点 p1 计算 d
        d = -np.dot(normal, p1)
        current_plane = (normal[0], normal[1], normal[2], d) # 当前平面模型

        # 3. 计算所有点到当前平面的距离，并找出内点
        # 距离 = |ax + by + cz + d| / sqrt(a^2+b^2+c^2)
        # 由于法向量已经归一化，分母为 1
        # 距离 = | points @ normal + d |  (@ 是矩阵乘法)
        distances = np.abs(np.dot(points, normal) + d)
        inlier_indices = np.where(distances < threshold)[0] # 找到距离小于阈值的点的索引
        num_inliers = len(inlier_indices)                   # 当前模型的内点数量

        # 4. 如果当前模型更好（内点更多），则更新最佳模型
        if num_inliers > max_inliers:
            # print(f"迭代 {i+1}: 找到新的最佳模型，内点数 = {num_inliers}")
            max_inliers = num_inliers
            best_inlier_indices = inlier_indices
            best_plane = current_plane

            # 可选：提前终止条件 (如果已经找到足够好的模型)
            # if num_inliers > num_points * 0.8: # 例如，如果超过 80% 的点是内点
            #    print(f"迭代 {i+1}: 找到足够好的模型 ({num_inliers} 内点)，提前终止。")
            #    break

    # 5. (可选但推荐) 使用找到的最佳内点集重新拟合平面，以获得更精确的模型
    if best_inlier_indices is not None and len(best_inlier_indices) >= min_inliers_count:
        print(f"RANSAC 结束。最佳模型包含 {len(best_inlier_indices)} 个内点。")
        print("使用所有内点进行平面优化...")
        inlier_points = points[best_inlier_indices]

        # 使用 PCA (主成分分析)，特别是对协方差矩阵进行 SVD (奇异值分解) 来找到最佳拟合平面
        # 平面的法向量对应于协方差矩阵的最小特征值对应的特征向量
        centroid = np.mean(inlier_points, axis=0)                 # 计算内点的质心
        centered_points = inlier_points - centroid                # 中心化内点
        covariance_matrix = np.cov(centered_points.T)             # 计算协方差矩阵 (需要转置)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix) # 对称矩阵使用 eigh

        # 最小特征值对应的特征向量即为优化后的法向量
        refined_normal = eigenvectors[:, np.argmin(eigenvalues)]
        # 使用质心重新计算 d
        refined_d = -np.dot(refined_normal, centroid)

        # 确保法向量方向一致性 (例如，如果可能，使其指向上方)
        # 这一步在后面计算旋转之前进行统一处理

        best_plane = (refined_normal[0], refined_normal[1], refined_normal[2], refined_d)
        # 根据优化后的平面重新计算最终的内点（可选，但可以得到更精确的内点集）
        distances = np.abs(np.dot(points, refined_normal) + refined_d)
        best_inlier_indices = np.where(distances < threshold)[0]
        print(f"优化后的模型最终内点数 = {len(best_inlier_indices)}。")

        return best_plane, best_inlier_indices

    else:
        print(f"RANSAC 失败：未能找到至少包含 {min_inliers_count} 个内点的平面。")
        return None, None


def get_rotation_matrix_to_align(v1, v2):
    """
    计算将向量 v1 旋转到与向量 v2 方向对齐的旋转矩阵 R。
    R * v1 ≈ (||v1|| / ||v2||) * v2  (方向对齐)

    Args:
        v1 (np.ndarray): 源向量 (3D)。
        v2 (np.ndarray): 目标向量 (3D)。

    Returns:
        np.ndarray: 3x3 的旋转矩阵。
    """
    # 归一化向量
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # 处理向量已经对齐或方向相反的特殊情况
    dot_product = np.dot(v1, v2)
    if np.isclose(dot_product, 1.0):
        # 向量已经对齐，无需旋转
        # print("向量已对齐，返回单位矩阵。")
        return np.identity(3)
    if np.isclose(dot_product, -1.0):
        # 向量方向相反。需要绕任意一个与 v1 垂直的轴旋转 180 度。
        # print("向量方向相反，计算 180 度旋转。")
        # 寻找一个与 v1 垂直的轴 (例如，与标准基向量叉乘)
        axis = np.cross(v1, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-6:  # 如果 v1 平行于 x 轴
            axis = np.cross(v1, [0, 1, 0])
        axis = axis / np.linalg.norm(axis) # 归一化旋转轴
        # 使用 Rodrigues 公式计算 180 度旋转 (theta=pi)
        # R = I + sin(pi)*K + (1-cos(pi))*K^2 = I + 0*K + 2*K^2 = I + 2*K^2
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]]) # 旋转轴的反对称矩阵
        return np.identity(3) + 2 * np.dot(K, K)


    # 一般情况：使用 Rodrigues 旋转公式
    axis = np.cross(v1, v2)             # 旋转轴是 v1 和 v2 的叉积
    axis = axis / np.linalg.norm(axis)  # 归一化旋转轴
    angle = np.arccos(dot_product)      # 旋转角度 (v1 和 v2 之间的夹角)

    # 旋转轴的反对称矩阵 K
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])

    # Rodrigues 旋转公式: R = I + sin(theta)*K + (1-cos(theta))*K^2
    R = np.identity(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R


def extract_ground_and_rotate(points, ransac_threshold=0.1, ransac_iterations=1000):
    """
    使用 RANSAC 提取地面平面，并旋转整个点云，使地面法向量接近 (0, 0, 1)。

    Args:
        points (np.ndarray): 输入点云，形状 (N, 3)。
        ransac_threshold (float): RANSAC 距离阈值。
        ransac_iterations (int): RANSAC 最大迭代次数。

    Returns:
        tuple: (rotated_points, ground_plane_params, ground_indices, rotation_matrix)
            - rotated_points (np.ndarray or None): 旋转后的点云，形状 (N, 3)。如果地面提取失败则为 None。
            - ground_plane_params (tuple or None): 检测到的地面平面参数 (a, b, c, d)。如果提取失败则为 None。
            - ground_indices (np.ndarray or None): 被识别为地面的点的索引。如果提取失败则为 None。
            - rotation_matrix (np.ndarray or None): 用于旋转的 3x3 矩阵。如果提取失败则为 None。
    """
    print("--- 开始地面平面提取 (RANSAC) ---")
    plane_params, inlier_indices = fit_plane_ransac(
        points,
        threshold=ransac_threshold,
        max_iterations=ransac_iterations,
        min_inliers_ratio=0.05 # 根据需要调整最小内点比例
    )

    if plane_params is None:
        print("地面平面提取失败。")
        return None, None, None, None

    a, b, c, d = plane_params
    normal = np.array([a, b, c])
    print(f"Original Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # --- 标准化法向量方向 ---
    # 确保法向量大致指向 "上方" (Z 分量为正)。
    # 这使得旋转目标 (0, 0, 1) 更加一致和稳定。
    if normal[2] < 0:
        print("检测到法向量 Z 分量为负，进行翻转使其指向上方。")
        normal = -normal
        d = -d
        # 更新平面参数以反映翻转
        plane_params = (normal[0], normal[1], normal[2], d)


    print(f"找到的地面法向量 (已定向): {normal}")
    target_normal = np.array([0.0, 0.0, 1.0]) # 目标法向量 (Z 轴正方向)

    print("\n--- 计算旋转矩阵 ---")
    rotation_matrix = get_rotation_matrix_to_align(normal, target_normal)
    print("计算得到的旋转矩阵 R:\n", rotation_matrix)

    # --- 验证旋转 (可选) ---
    rotated_normal_check = rotation_matrix @ normal # R * n
    print(f"原始法向量: {normal}")
    print(f"旋转后的法向量 (应接近 {target_normal}): {rotated_normal_check}")
    # 由于浮点数计算，结果可能不会完全精确等于 [0, 0, 1]，但应该非常接近

    print("\n--- 对点云应用旋转 ---")
    # 应用旋转: rotated_p = R * p
    # 对于存储为行向量的 N x 3 点云矩阵 P:
    # rotated_P = (R @ P.T).T  或者等效地: rotated_P = P @ R.T
    # 我们将使用 P @ R.T 的形式
    rotated_points = points @ rotation_matrix.T
    # 或者使用 np.dot: rotated_points = np.dot(points, rotation_matrix.T)

    print("点云旋转完成。")

    return rotated_points, plane_params, inlier_indices, rotation_matrix
