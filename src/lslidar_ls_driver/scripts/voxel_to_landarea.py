# 导入必要的库
import open3d as o3d
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from collections import deque # 用于BFS的队列

# ... (load_point_clouds_from_folder, voxelize_point_cloud,
#      create_mesh_from_voxel_grid_colored_by_z, visualize_geometry,
#      denoise_point_cloud_sor, extract_ground_RANSAC_NormalVectorConstraint 函数与之前相同，此处省略) ...
def load_point_clouds_from_folder(folder_path, file_extensions=('.ply', '.pcd', '.xyz')):
    """
    从指定文件夹加载所有支持格式的点云文件，并合并它们。
    (函数内容与之前相同)
    """
    all_points_list = []
    print(f"正在从文件夹 '{folder_path}' 加载点云...")
    for ext in file_extensions:
        search_pattern = os.path.join(folder_path, f"*{ext}")
        file_paths = glob.glob(search_pattern)
        for file_path in file_paths:
            try:
                print(f"  正在读取: {file_path}")
                pcd = o3d.io.read_point_cloud(file_path)
                if pcd.has_points():
                    all_points_list.append(pcd)
                else:
                    print(f"  警告: 文件 {file_path} 为空或无法读取点。")
            except Exception as e:
                print(f"  错误: 读取文件 {file_path} 失败: {e}")

    if not all_points_list:
        print("错误: 在指定文件夹中未找到有效的点云文件。")
        return None

    if len(all_points_list) == 1:
        print(f"总共加载了 {len(all_points_list[0].points)} 个点。")
        return all_points_list[0]
    else:
        merged_pcd = o3d.geometry.PointCloud()
        for pcd_part in all_points_list:
            merged_pcd += pcd_part
        print(f"总共加载并合并了 {len(merged_pcd.points)} 个点 (来自 {len(all_points_list)} 个文件)。")
        return merged_pcd

def voxelize_point_cloud(point_cloud, voxel_size):
    """
    将点云进行体素化。
    (函数内容与之前相同)
    """
    if not point_cloud or not point_cloud.has_points():
        print("错误: 输入的点云为空，无法进行体素化。")
        return None

    print(f"正在以 {voxel_size} 的体素大小对点云进行体素化...")
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        point_cloud,
        voxel_size=voxel_size
    )
    print(f"体素化完成，生成了 {len(voxel_grid.get_voxels())} 个占用的体素。")
    return voxel_grid

def create_mesh_from_voxel_grid_colored_by_z(voxel_grid, colormap_name='terrain'):
    """
    从体素栅格创建一个由立方体组成的网格，并根据Z值（高程）对其进行着色。
    (函数内容与之前相同)
    """
    if not voxel_grid or not voxel_grid.get_voxels():
        print("错误: 体素栅格为空，无法创建网格。")
        return None

    print(f"正在从体素栅格创建网格，并根据Z值使用 '{colormap_name}' 颜色映射进行着色...")

    voxels = voxel_grid.get_voxels()
    voxel_size = voxel_grid.voxel_size

    z_coords = []
    for voxel in voxels:
        center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
        z_coords.append(center[2])

    if not z_coords:
        print("警告: 体素栅格中没有有效的体素坐标。")
        return None

    min_z = np.min(z_coords)
    max_z = np.max(z_coords)
    z_range = max_z - min_z
    print(f"  体素Z值范围: [{min_z:.2f}, {max_z:.2f}]")

    if z_range == 0:
        z_range = 1.0

    try:
        cmap = plt.get_cmap(colormap_name)
    except ValueError:
        print(f"警告: 颜色映射 '{colormap_name}' 未找到，将使用默认的 'viridis'。")
        cmap = plt.get_cmap('viridis')

    combined_mesh = o3d.geometry.TriangleMesh()

    for i, voxel in enumerate(voxels):
        center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
        current_z = center[2]

        normalized_z = (current_z - min_z) / z_range if z_range > 0 else 0.5

        color_rgba = cmap(normalized_z)
        rgb_color = [color_rgba[0], color_rgba[1], color_rgba[2]]

        cube = o3d.geometry.TriangleMesh.create_box(
            width=voxel_size,
            height=voxel_size,
            depth=voxel_size
        )
        cube.paint_uniform_color(rgb_color)

        translation_vector = center - np.array([voxel_size / 2, voxel_size / 2, voxel_size / 2])
        cube.translate(translation_vector)

        combined_mesh += cube

        if (i + 1) % 1000 == 0:
            print(f"  已处理 {i + 1}/{len(voxels)} 个体素...")

    print(f"网格创建完成，包含 {len(combined_mesh.triangles)} 个三角形。")
    return combined_mesh

def visualize_geometry(geometry_list, window_name="Open3D 可视化"):
    """
    可视化一个或多个Open3D几何对象。
    (函数内容与之前相同)
    """
    if not geometry_list:
        print("错误: 没有可供可视化的几何对象。")
        return
    print(f"正在启动可视化窗口: {window_name}...")
    o3d.visualization.draw_geometries(geometry_list, window_name=window_name)
    print("可视化窗口已关闭。")

def denoise_point_cloud_sor(point_cloud, nb_neighbors=20, std_ratio=2.0, show_comparison=False):
    """
    使用 Statistical Outlier Removal (SOR) 对点云进行去噪。
    (函数内容与之前相同)
    """
    if not point_cloud or not point_cloud.has_points():
        print("错误: 输入的点云为空，无法进行去噪。")
        return None

    print(f"正在对点云进行 Statistical Outlier Removal (SOR) 去噪...")
    print(f"  参数: nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")
    print(f"  去噪前点数: {len(point_cloud.points)}")

    cl, ind = point_cloud.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )

    denoised_pcd = point_cloud.select_by_index(ind)

    num_removed = len(point_cloud.points) - len(denoised_pcd.points)
    print(f"  去噪后点数: {len(denoised_pcd.points)} (移除了 {num_removed} 个离群点)")

    if show_comparison:
        print("  显示去噪前后的对比 (蓝色: 原始, 红色: 去噪后)...")
        original_pcd_colored = o3d.geometry.PointCloud(point_cloud)
        original_pcd_colored.paint_uniform_color([0, 0.651, 0.929]) # 蓝色

        denoised_pcd_colored = o3d.geometry.PointCloud(denoised_pcd)
        denoised_pcd_colored.paint_uniform_color([1, 0, 0]) # 红色

        o3d.visualization.draw_geometries(
            [original_pcd_colored, denoised_pcd_colored],
            window_name="SOR 去噪对比 (蓝:原始, 红:去噪后)"
        )

    return denoised_pcd

def extract_ground_RANSAC_NormalVectorConstraint(point_cloud, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """
    使用 RANSAC 算法从点云中提取地面，并旋转点云使地面法向量接近（0, 0, 1）
    (函数内容与之前相同，做了微小调整以确保旋转后主平面法向为正Z)
    """
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=distance_threshold,
                                                     ransac_n=ransac_n,
                                                     num_iterations=num_iterations)
    [a, b, c, d] = plane_model
    print(f"Original Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # 确保法向量朝向Z轴正方向 (通常地面在下方，法线朝上)
    if c < 0:
        a, b, c, d = -a, -b, -c, -d
        
    current_normal = np.array([a, b, c])
    target_normal = np.array([0, 0, 1])

    # 如果法向量已经非常接近目标法向量，则跳过旋转
    if np.allclose(current_normal, target_normal, atol=1e-3) or \
       np.allclose(current_normal, -target_normal, atol=1e-3) : # 也考虑方向完全相反的情况
        print("地面法向量已接近Z轴，跳过旋转。")
        # 如果方向相反，但我们希望是(0,0,1)，还是需要处理
        if np.allclose(current_normal, -target_normal, atol=1e-3) and target_normal[2] > 0:
             # 创建一个绕X轴旋转180度的矩阵
            R_180_x = o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, 0))
            point_cloud.rotate(R_180_x, center=point_cloud.get_center())
            print("  (已将法向量翻转至Z轴正向)")
    else:
        axis = np.cross(current_normal, target_normal)
        if np.linalg.norm(axis) < 1e-6: # 法向量平行或反平行
            if np.dot(current_normal, target_normal) < 0: # 反平行
                # 找一个与current_normal垂直的轴进行180度旋转
                # 例如，如果current_normal是(0,0,-1)，target是(0,0,1)，叉积为0
                # 可以选择绕x轴旋转180度
                axis = np.array([1, 0, 0]) if abs(current_normal[0]) < 0.9 else np.array([0, 1, 0]) # 避免与法向量平行
            else: # 平行且同向，无需旋转
                axis = np.array([0,0,0]) # 无需旋转的标志
        
        if np.linalg.norm(axis) > 1e-6: # 仅当需要旋转时执行
            axis = axis / np.linalg.norm(axis)
            theta = np.arccos(np.dot(current_normal, target_normal) /
                              (np.linalg.norm(current_normal) * np.linalg.norm(target_normal)))

            def rodrigues_rotation(axis_rot, theta_rot):
                K = np.array([
                    [0, -axis_rot[2], axis_rot[1]],
                    [axis_rot[2], 0, -axis_rot[0]],
                    [-axis_rot[1], axis_rot[0], 0]
                ])
                R = np.eye(3) + np.sin(theta_rot) * K + (1 - np.cos(theta_rot)) * np.dot(K, K)
                return R

            rotation_matrix = rodrigues_rotation(axis, theta)
            print("旋转矩阵:\n", rotation_matrix)
            point_cloud.rotate(rotation_matrix, center=point_cloud.get_center())


    # 重新计算地面平面方程 (可选, 用于验证)
    # rotated_inliers = point_cloud.select_by_index(inliers)
    # rotated_plane_model, _ = rotated_inliers.segment_plane(distance_threshold=distance_threshold,
    #                                                        ransac_n=ransac_n,
    #                                                        num_iterations=num_iterations)
    # [a_rot, b_rot, c_rot, d_rot] = rotated_plane_model
    # print(f"Rotated Plane equation: {a_rot:.2f}x + {b_rot:.2f}y + {c_rot:.2f}z + {d_rot:.2f} = 0")

    # # 不再在这里上色，让后续步骤处理
    # inlier_cloud = point_cloud.select_by_index(inliers)
    # inlier_cloud.paint_uniform_color([1.0, 0, 0])
    # outlier_cloud = point_cloud.select_by_index(inliers, invert=True)
    # outlier_cloud.paint_uniform_color([0, 1.0, 0])
    return point_cloud

# --- 新增：候选着陆区提取函数 ---
def find_candidate_landing_zones(voxel_grid,
                                 min_voxels_for_area,
                                 max_local_z_diff,
                                 max_overall_slope_degrees,
                                 ransac_dist_thresh_slope=0.1,
                                 ransac_n_slope=3,
                                 ransac_iter_slope=100):
    """
    从体素栅格中提取候选着陆区域。

    参数:
    voxel_grid (open3d.geometry.VoxelGrid): 输入的体素栅格。
    min_voxels_for_area (int): 构成有效区域所需的最小体素数量。
                               实际面积 = min_voxels_for_area * voxel_size^2。
    max_local_z_diff (float): 相邻体素中心Z值的最大允许差，用于判断局部平坦度。
                              建议值为 voxel_size 的倍数，例如 0.5 * voxel_size。
    max_overall_slope_degrees (float): 候选区域整体平面的最大允许坡度（度）。
    ransac_dist_thresh_slope (float): RANSAC平面拟合时点到平面的距离阈值。
    ransac_n_slope (int): RANSAC平面拟合时选择的点数。
    ransac_iter_slope (int): RANSAC平面拟合的迭代次数。

    返回:
    list[open3d.geometry.TriangleMesh]: 一个包含所有有效候选着陆区网格的列表。
                                        每个网格由构成该区域的体素立方体组成，并着色为红色。
    """
    if not voxel_grid or not voxel_grid.get_voxels():
        print("错误: 体素栅格为空，无法提取着陆区。")
        return []

    print("开始提取候选着陆区域...")
    voxels_data = voxel_grid.get_voxels() # 获取所有体素 (o3d.geometry.Voxel)
    voxel_size = voxel_grid.voxel_size
    
    # 将体素数据转换为更易于处理的格式：(grid_index_tuple: center_coords)
    # 并创建一个 grid_index 到 Voxel 对象的映射，方便后续查找
    voxel_map = {} # grid_index_tuple -> o3d.geometry.Voxel
    voxel_centers_map = {} # grid_index_tuple -> center_np_array
    
    for v_obj in voxels_data:
        idx_tuple = tuple(v_obj.grid_index)
        voxel_map[idx_tuple] = v_obj
        voxel_centers_map[idx_tuple] = voxel_grid.get_voxel_center_coordinate(v_obj.grid_index)

    visited_indices = set()
    candidate_landing_zones_meshes = []
    zone_count = 0

    # 定义6个方向的邻居偏移 (dx, dy, dz)
    # 我们只关心XY平面上的邻居，因为高度由max_local_z_diff控制
    neighbor_offsets = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        # 如果也考虑对角线邻居，可以添加 (1,1,0), (1,-1,0) 等
        # (1, 0, 1), (1, 0, -1) 等是检查上下邻居，但我们用z_diff控制
    ]

    for start_voxel_obj in voxels_data:
        start_idx = tuple(start_voxel_obj.grid_index)
        if start_idx in visited_indices:
            continue

        current_region_indices = [] # 存储当前连通区域的体素索引
        queue = deque([start_idx])
        visited_indices.add(start_idx)
        current_region_indices.append(start_idx)
        
        start_voxel_center = voxel_centers_map[start_idx]

        # 1. 使用BFS进行区域生长 (检查局部平坦度)
        head = 0
        while head < len(current_region_indices): # 用列表模拟队列，或者直接用deque
            current_idx = current_region_indices[head]
            head += 1
            current_center = voxel_centers_map[current_idx]

            for offset in neighbor_offsets:
                # 我们主要关心XY平面上的邻居，Z方向的平坦度由max_local_z_diff控制
                # 所以我们查找的是 (ix+dx, iy+dy, iz_neighbor) 这样的邻居
                # 一个简化的方法是，先找到XY邻居的grid_index，然后检查其Z值是否接近
                
                # 查找严格的XY平面邻居 (ix+dx, iy+dy, iz)
                # 然后检查这个假定邻居的真实Z值与当前体素Z值的差异
                potential_neighbor_xy_idx = (current_idx[0] + offset[0], current_idx[1] + offset[1])
                
                # 迭代检查 potential_neighbor_xy_idx 在不同Z层是否有体素
                # 通常，地形起伏不会太大，检查当前Z层和上下Z层就够了
                for dz_offset in [-1, 0, 1]: # 检查当前层和紧邻的上下层
                    neighbor_idx_check = (potential_neighbor_xy_idx[0], potential_neighbor_xy_idx[1], current_idx[2] + dz_offset)

                    if neighbor_idx_check in voxel_map and neighbor_idx_check not in visited_indices:
                        neighbor_center = voxel_centers_map[neighbor_idx_check]
                        # 局部平坦度检查：Z值差异
                        if abs(neighbor_center[2] - current_center[2]) <= max_local_z_diff:
                            visited_indices.add(neighbor_idx_check)
                            current_region_indices.append(neighbor_idx_check)
        
        # 2. 检查有效面积
        num_voxels_in_region = len(current_region_indices)
        area_m2 = num_voxels_in_region * (voxel_size ** 2)
        
        if num_voxels_in_region < min_voxels_for_area:
            # print(f"  区域太小 ({num_voxels_in_region}个体素), 跳过.")
            continue
        
        print(f"  发现一个潜在区域，包含 {num_voxels_in_region} 个体素 (面积约 {area_m2:.2f} m^2)。")

        # 3. 检查整体坡度
        region_points = np.array([voxel_centers_map[idx] for idx in current_region_indices])
        
        if len(region_points) < ransac_n_slope : # RANSAC拟合平面至少需要 ransac_n_slope 个点
            print(f"    区域点数 ({len(region_points)}) 不足以进行坡度分析，跳过。")
            continue

        # 用这些体素中心点创建一个临时点云以进行平面拟合
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(region_points)
        
        # 使用RANSAC拟合平面
        try:
            plane_model, _ = temp_pcd.segment_plane(distance_threshold=ransac_dist_thresh_slope,
                                                    ransac_n=ransac_n_slope,
                                                    num_iterations=ransac_iter_slope)
        except RuntimeError as e:
            print(f"    RANSAC平面拟合失败: {e}, 跳过此区域。可能是点共线或共面问题。")
            continue

        # plane_model: [a, b, c, d] for ax + by + cz + d = 0
        # 法向量为 (a, b, c)
        normal_vector = np.array(plane_model[:3])
        normal_vector_normalized = normal_vector / np.linalg.norm(normal_vector)
        
        # 确保法向量朝上 (c > 0). 如果地面是水平的，法向量应该是 (0,0,1) 或 (0,0,-1)
        # 经过之前的 extract_ground_RANSAC_NormalVectorConstraint，我们期望主平面法向量是 (0,0,1)
        if normal_vector_normalized[2] < 0:
            normal_vector_normalized = -normal_vector_normalized
            
        # Z轴单位向量
        z_axis = np.array([0, 0, 1])
        
        # 计算法向量与Z轴的夹角（即坡度）
        # cos_angle = np.dot(normal_vector_normalized, z_axis)
        # 由于normal_vector_normalized[2]已经是cos(angle) (如果normal_vector_normalized指向+Z)
        # 或者更准确地，cos_angle = |normal_vector_normalized[2]| 因为我们关心与XY平面的夹角
        cos_angle = abs(normal_vector_normalized[2]) # abs确保处理法向量可能朝下的情况（虽然我们已经翻转了）
        
        # 防止 acos 的参数超出 [-1, 1] 范围 (由于浮点数精度问题)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        slope_radians = np.arccos(cos_angle)
        slope_degrees = np.degrees(slope_radians)

        print(f"    区域整体坡度: {slope_degrees:.2f} 度。")

        if slope_degrees <= max_overall_slope_degrees:
            zone_count += 1
            print(f"  ---> 找到第 {zone_count} 个有效着陆区！面积: {area_m2:.2f} m^2, 坡度: {slope_degrees:.2f}°")
            # 为这个着陆区创建一个网格并上色
            landing_zone_mesh = o3d.geometry.TriangleMesh()
            for voxel_idx in current_region_indices:
                center = voxel_centers_map[voxel_idx]
                cube = o3d.geometry.TriangleMesh.create_box(width=voxel_size, height=voxel_size, depth=voxel_size)
                cube.paint_uniform_color([1, 0, 0]) # 红色表示有效着陆区
                cube.translate(center - np.array([voxel_size / 2, voxel_size / 2, voxel_size / 2]))
                landing_zone_mesh += cube
            candidate_landing_zones_meshes.append(landing_zone_mesh)
        else:
            print(f"    区域坡度 ({slope_degrees:.2f}°) 过大，不合格。")

    if not candidate_landing_zones_meshes:
        print("未找到满足所有条件的候选着陆区域。")
    else:
        print(f"总共找到 {len(candidate_landing_zones_meshes)} 个候选着陆区域。")
        
    return candidate_landing_zones_meshes

# --- 主程序 ---
if __name__ == "__main__":
    # 1. 设置点云文件夹路径
    demo_folder = r'./data/voxel'
    # 如果文件夹不存在，可以取消下面这行注释来创建演示数据
    # os.makedirs(demo_folder, exist_ok=True)
    # demo_file_path_ply = os.path.join(demo_folder, "sample_terrain_noisy.ply")
    
    # # 创建一个类似地形的点云 (如果需要演示数据)
    # points = []
    # num_x, num_y = 50, 50
    # for i in range(num_x):
    #     for j in range(num_y):
    #         x = i * 0.2 - (num_x * 0.2 / 2)
    #         y = j * 0.2 - (num_y * 0.2 / 2)
    #         z = np.sin(np.sqrt(x**2 + y**2) / 2) * 0.8 + np.random.rand() * 0.1 # 地形起伏减小
    #         points.append([x, y, z])
    #         # 添加少量高斯噪声点
    #         if np.random.rand() < 0.01:
    #             points.append([x + np.random.normal(0,0.5), y + np.random.normal(0,0.5), z + np.random.normal(0,1.5)])
    
    # if points:
    #     pcd_demo = o3d.geometry.PointCloud()
    #     pcd_demo.points = o3d.utility.Vector3dVector(np.array(points))
    #     o3d.io.write_point_cloud(demo_file_path_ply, pcd_demo)
    #     print(f"已在 '{demo_folder}' 文件夹中创建带噪声的演示点云文件 '{os.path.basename(demo_file_path_ply)}'")
    
    point_cloud_folder = demo_folder
    
    # 2. 加载点云
    original_pcd = load_point_clouds_from_folder(point_cloud_folder)

    if original_pcd:
        # 可选：对原始点云进行人工旋转以模拟倾斜地面 (如果需要)
        # print("对原始点云进行人工旋转...")
        # rotation_degrees_x = -10 # 绕X轴旋转
        # rotation_radians_x = np.radians(rotation_degrees_x)
        # rotation_matrix_x = o3d.geometry.get_rotation_matrix_from_axis_angle([rotation_radians_x, 0, 0])
        # original_pcd.rotate(rotation_matrix_x, center=original_pcd.get_center())
        
        # rotation_degrees_y = 5 # 绕Y轴旋转
        # rotation_radians_y = np.radians(rotation_degrees_y)
        # rotation_matrix_y = o3d.geometry.get_rotation_matrix_from_axis_angle([0, rotation_radians_y, 0])
        # original_pcd.rotate(rotation_matrix_y, center=original_pcd.get_center())
        # print("人工旋转完成。")


        # RANSAC提取地面点云，并旋转使主平面与XY平面平行
        # 调整 distance_threshold 以适应您的数据，如果点云比较稀疏或噪声大，可能需要更大的值
        ground_align_distance_thresh = 0.2 # 例如0.2米
        print(f"使用RANSAC校正地面，阈值: {ground_align_distance_thresh}")
        original_pcd = extract_ground_RANSAC_NormalVectorConstraint(original_pcd, 
                                                                    distance_threshold=ground_align_distance_thresh,
                                                                    num_iterations=2000) # 增加迭代次数

        print("可视化校正地面后的点云...")
        visualize_geometry([original_pcd], window_name="校正地面后的点云")

        # 对点云进行去噪
        denoised_pcd = denoise_point_cloud_sor(
            original_pcd,
            nb_neighbors=50,   # 邻居点数量, 根据点云密度调整
            std_ratio=1.0,     # 标准差比例阈值, 越小越严格
            show_comparison=False
        )

        if not denoised_pcd or not denoised_pcd.has_points():
            print("错误: 去噪后点云为空，程序终止。")
            exit()
            
        # visualize_geometry([denoised_pcd], window_name="去噪后的点云")

        # 3. 设置体素化参数
        chosen_voxel_size = 0.5 # 体素大小，单位：米

        # 4. 使用去噪后的点云进行体素化
        voxel_grid_result = voxelize_point_cloud(denoised_pcd, voxel_size=chosen_voxel_size)

        if voxel_grid_result:
            # 可视化原始体素网格 (Z值着色)
            voxel_mesh_z_colored = create_mesh_from_voxel_grid_colored_by_z(
                voxel_grid_result, 
                colormap_name='viridis' # 使用 'terrain' 或 'jet'
            )
            if voxel_mesh_z_colored:
                 visualize_geometry([voxel_mesh_z_colored],
                                   window_name=f"体素网格 (Z值着色, Voxel: {chosen_voxel_size})")


            # --- 新增：提取候选着陆区 ---
            # 参数设置 (根据实际需求调整)
            min_landing_area_m2 = 2.0  # 最小着陆面积 (平方米)
            # min_voxels_for_area: 实际面积 = min_voxels_for_area * voxel_size^2
            min_voxels_count = int(min_landing_area_m2 / (chosen_voxel_size ** 2))
            
            # max_local_z_diff: 相邻体素Z值最大差，用于判断局部平坦度
            # 建议值: 0.5 * chosen_voxel_size 到 1.0 * chosen_voxel_size
            # 如果地形起伏较大，或者希望找到更平坦的区域，可以减小此值
            # 如果 chosen_voxel_size 较大，此值也应相应增大
            local_flatness_threshold = chosen_voxel_size * 0.75 
                                       
            max_slope_deg = 10.0        # 区域整体最大坡度 (度)

            print(f"\n--- 开始寻找着陆区 ---")
            print(f"参数: 体素大小={chosen_voxel_size}m, 最小面积={min_landing_area_m2}m^2 ({min_voxels_count}个体素), "
                  f"局部Z差阈值={local_flatness_threshold:.3f}m, 最大坡度={max_slope_deg}°")

            candidate_zones = find_candidate_landing_zones(
                voxel_grid=voxel_grid_result,
                min_voxels_for_area=min_voxels_count,
                max_local_z_diff=local_flatness_threshold,
                max_overall_slope_degrees=max_slope_deg,
                ransac_dist_thresh_slope=chosen_voxel_size * 1, # RANSAC距离阈值，与体素大小相关
                ransac_n_slope=5, # 拟合平面至少需要5个体素点
                ransac_iter_slope=100
            )
            
            # 可视化结果
            if candidate_zones:
                print(f"可视化找到的 {len(candidate_zones)} 个候选着陆区域...")
                # 可以将原始Z值着色的体素网格和着陆区一起显示
                # 为了更好地区分，可以给原始网格一个半透明效果，但这需要更复杂的渲染设置
                # 这里简单地将它们一起显示
                display_list = []
                if voxel_mesh_z_colored: # 添加原始彩色体素网格作为背景
                    display_list.append(voxel_mesh_z_colored)
                display_list.extend(candidate_zones) # 添加红色着陆区
                
                visualize_geometry(display_list,
                                   window_name="候选着陆区域 (红色) 与 原始体素网格")
            else:
                print("未找到可显示的候选着陆区域。")
            # --- 提取结束 ---

        else:
            print("未能成功进行体素化。")
    else:
        print("未能加载点云数据，程序终止。")