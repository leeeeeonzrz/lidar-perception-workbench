import numpy as np
from collections import defaultdict
import time
import open3d as o3d
from scipy.ndimage import binary_erosion
from multiprocessing import Event
from scipy.ndimage import binary_erosion, label


class LandingZoneDetector():
    """
    优化版着陆区域检测器
    """

    def __init__(self):
        # 参数设置
        self.grid_resolution = 2
        self.landing_zone_radius = 8
        self.max_cell_std_dev = 1  #0.08
        self.max_cell_range = 1  #0.15
        self.max_zone_planarity_std_dev = 1 #0.10
        self.min_points_per_cell = 4

        # 预计算圆形掩模
        self.cells_in_radius = int(np.ceil(self.landing_zone_radius / self.grid_resolution))  # 半径对应的格子数
        self.circular_mask = self._create_circular_mask(self.cells_in_radius) # 创建圆形掩模

        # 内部状态
        self.grid_stats = {}  # 存储栅格统计数据
        self.best_landing_zone = None
        self.running = True
        self.pcd_queue = None

        # 预分配数组
        self.flat_cells_indices = None
        self.cell_centers = None

    def _create_circular_mask(self, radius):
        """创建圆形掩模用于快速检查圆形区域"""
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        mask = x ** 2 + y ** 2 <= radius ** 2
        return mask

    def spinOnce0(self, pcd, stop_evt:Event = None):
        """单次处理"""
        start_time = time.time()

        if pcd.has_points():

            # 创建2.5D栅格统计地图
            self._create_25d_grid_stats(pcd)

            end_time1 = time.time()
            elapsed_time1 = end_time1 - start_time
            # print(f"创建栅格运行时间: {elapsed_time1:.4f} 秒")

            # 搜索最佳着陆区
            self.best_landing_zone = self._find_best_landing_zone()

            end_time2 = time.time()
            elapsed_time2 = end_time2 - end_time1
            # print(f"搜索着陆区时间: {elapsed_time2:.4f} 秒")
            # print(f"着陆区总时间: {end_time2-start_time:.4f} 秒")

            elapsed_time_ms = (end_time2 - start_time) * 1000
            # print(f"代码运行时间: {elapsed_time:.4f} 秒")
            target_duration_ms = 70.0  # 目标运行时间（毫秒）
            remaining_time_ms = target_duration_ms - elapsed_time_ms

            # 只有当实际运行时间小于目标时间时才需要等待
            if remaining_time_ms > 0:
                # print(f"算法运行了 {elapsed_time_ms:.2f} ms, 开始精确等待 {remaining_time_ms:.2f} ms")
                wait_start = time.perf_counter()  # 使用更高精度的计时器

                # 忙等待循环
                while (time.perf_counter() - wait_start) * 1000 < remaining_time_ms:
                    # 1. 检查退出事件，这是最重要的！
                    if stop_evt and stop_evt.is_set():
                        print("[spinOnce] 收到停止信号，提前退出等待。")
                        return  # 直接返回，结束函数

                    # 2. 加入一个极短的sleep，避免100%占用CPU
                    #    这是一个很好的平衡，既保证了响应速度，又不会让CPU风扇狂转
                    time.sleep(0.001)  # 等待1毫秒
            # else:
            #     print(f"算法运行时间 {elapsed_time_ms:.2f} ms 已超过目标时间 {target_duration_ms} ms")

            # 最终总时间会非常接近95ms
            # final_time = time.time()
            # final_elapsed = (final_time - start_time) * 1000
            # print(f"着陆总运行时间: {final_elapsed:.2f} ms")

    def spinOnce(self, pcd, stop_evt: Event = None):
        """单次处理"""
        start_time = time.time()

        if pcd.has_points():
            points_np = np.asarray(pcd.points)
            # --- 核心修改：分流逻辑 ---
            # 1. 计算点云中心 (x, y, z)
            center = np.mean(points_np, axis=0)
            # 2. 计算中心点到原点 (0,0,0) 的 3D 欧几里得距离
            distance = np.linalg.norm(center)
            # distance = np.abs(center[1])

            if distance > 30.0:

                # 创建2.5D栅格统计地图
                self._create_25d_grid_stats(pcd)

                end_time1 = time.time()
                elapsed_time1 = end_time1 - start_time
                # print(f"创建栅格运行时间: {elapsed_time1:.4f} 秒")

                # 搜索最佳着陆区
                self.best_landing_zone = self._find_best_landing_zone()

            end_time2 = time.time()
            # elapsed_time2 = end_time2 - end_time1
            # print(f"搜索着陆区时间: {elapsed_time2:.4f} 秒")
            # print(f"着陆区总时间: {end_time2-start_time:.4f} 秒")

            elapsed_time_ms = (end_time2 - start_time) * 1000
            # print(f"代码运行时间: {elapsed_time:.4f} 秒")
            target_duration_ms = 70.0  # 目标运行时间（毫秒）
            remaining_time_ms = target_duration_ms - elapsed_time_ms

            # 只有当实际运行时间小于目标时间时才需要等待
            if remaining_time_ms > 0:
                # print(f"算法运行了 {elapsed_time_ms:.2f} ms, 开始精确等待 {remaining_time_ms:.2f} ms")
                wait_start = time.perf_counter()  # 使用更高精度的计时器

                # 忙等待循环
                while (time.perf_counter() - wait_start) * 1000 < remaining_time_ms:
                    # 1. 检查退出事件，这是最重要的！
                    if stop_evt and stop_evt.is_set():
                        print("[spinOnce] 收到停止信号，提前退出等待。")
                        return  # 直接返回，结束函数

                    # 2. 加入一个极短的sleep，避免100%占用CPU
                    #    这是一个很好的平衡，既保证了响应速度，又不会让CPU风扇狂转
                    time.sleep(0.001)  # 等待1毫秒
            # else:
            #     print(f"算法运行时间 {elapsed_time_ms:.2f} ms 已超过目标时间 {target_duration_ms} ms")

            # 最终总时间会非常接近95ms
            # final_time = time.time()
            # final_elapsed = (final_time - start_time) * 1000
            # print(f"着陆总运行时间: {final_elapsed:.2f} ms")

    def _create_25d_grid_stats(self, pcd:o3d.geometry.PointCloud):
        self.grid_stats.clear()

        # print(f"原始点云数量: {len(pcd.points)}")
        # 体素下采样
        downpcd = pcd.voxel_down_sample(voxel_size=0.8)
        points = np.asarray(downpcd.points)
        # print(f"下采样后点云数量: {len(points)}")

        # start_time = time.time()

        # 1. 计算栅格索引
        indices = np.floor(points[:, :2] / self.grid_resolution).astype(int)

        # 2. 为每个点计算一个唯一的线性索引，便于排序和分组
        # 找到索引的最小值，用于将二维索引转换为一维
        ix_min, iy_min = indices.min(axis=0)

        # 将二维索引 (ix, iy) 转换为一维整数键
        linear_indices = (indices[:, 0] - ix_min) * (indices[:, 1].max() - iy_min + 1) + (indices[:, 1] - iy_min)

        # 3. 使用 np.unique 进行高效分组
        # np.unique 的 return_inverse=True 会返回一个数组，标记每个原始元素在唯一值数组中的位置
        # 这实际上就是我们的分组标签
        unique_keys, group_labels = np.unique(linear_indices, return_inverse=True)

        # 准备用于聚合的Z值
        z_values = points[:, 2]

        # 4. 使用 np.bincount 进行超快聚合
        # np.bincount 可以根据权重对整数标签进行求和、计数等操作

        # 计算每个组的点数
        counts = np.bincount(group_labels)

        # 计算每个组的Z值总和
        sum_z = np.bincount(group_labels, weights=z_values)

        # 计算每个组的Z值平方和 (用于计算标准差)
        sum_z_sq = np.bincount(group_labels, weights=z_values ** 2)

        # 计算每个组的Z值最小值和最大值
        # 我们按组标签排序Z值，然后找到每个组的边界
        sorted_z = z_values[np.argsort(group_labels)]
        # 使用 np.diff 和 np.where 找到分组的边界
        group_boundaries = np.where(np.diff(group_labels[np.argsort(group_labels)]))[0] + 1
        # 在开头和结尾添加边界，以便处理所有组
        boundaries = np.concatenate(([0], group_boundaries, [len(z_values)]))

        min_z = np.minimum.reduceat(sorted_z, boundaries[:-1])
        max_z = np.maximum.reduceat(sorted_z, boundaries[:-1])

        # 5. 计算最终统计量
        mean_z = sum_z / counts
        variance = (sum_z_sq / counts) - (mean_z ** 2)
        # 处理数值稳定性问题，避免出现微小的负数方差
        variance[variance < 0] = 0
        std_dev_z = np.sqrt(variance)
        range_z = max_z - min_z

        # end_time_stats = time.time()
        # print(f"NumPy 统计时间: {end_time_stats - start_time:.4f} 秒")

        # 6. 过滤并构建结果字典
        # 首先过滤掉点数不足的单元格
        valid_mask = counts >= self.min_points_per_cell

        # 只保留有效单元格的键和统计值
        valid_keys = unique_keys[valid_mask]
        valid_counts = counts[valid_mask]
        valid_min_z = min_z[valid_mask]
        valid_max_z = max_z[valid_mask]
        valid_mean_z = mean_z[valid_mask]
        valid_std_dev_z = std_dev_z[valid_mask]
        valid_range_z = range_z[valid_mask]

        # 将一维键转换回二维索引
        num_cols = indices[:, 1].max() - iy_min + 1
        valid_ix = (valid_keys // num_cols) + ix_min
        valid_iy = (valid_keys % num_cols) + iy_min

        # 使用 zip 快速构建字典
        self.grid_stats = {
            (ix, iy): {
                'count': int(c),
                'min_z': float(mn_z),
                'max_z': float(mx_z),
                'avg_z': float(avg_z),
                'std_dev_z': float(std_z),
                'range_z': float(r_z)
            }
            for ix, iy, c, mn_z, mx_z, avg_z, std_z, r_z in zip(
                valid_ix, valid_iy, valid_counts, valid_min_z, valid_max_z,
                valid_mean_z, valid_std_dev_z, valid_range_z
            )
        }

        # end_time_loop = time.time()
        # print(f"构建字典时间: {end_time_loop - start_time:.4f} 秒")


    def _find_best_landing_zone(self):
        """
        使用NumPy向量化操作在统计栅格中搜索最佳着陆区。
        """
        if not self.grid_stats:
            return None

        # --- 步骤 1: 创建平坦性布尔矩阵 ---
        # 首先找出所有栅格索引的范围，以创建一个足够大的矩阵
        all_indices = np.array(list(self.grid_stats.keys()))
        if all_indices.size == 0:
            return None

        min_ix, min_iy = np.min(all_indices, axis=0)
        max_ix, max_iy = np.max(all_indices, axis=0)

        # 创建一个布尔矩阵，初始为False。True表示该单元格是平坦的。
        # 我们使用 min/max 来创建一个紧凑的矩阵，而不是一个可能巨大的稀疏矩阵。
        # 注意：这里需要处理负索引，所以我们进行偏移。
        # 矩阵的坐标是 (iy, ix) 对应数组的 [row, col]
        rows = max_iy - min_iy + 1
        cols = max_ix - min_ix + 1
        is_flat = np.zeros((rows, cols), dtype=bool)

        # 填充平坦性矩阵
        for (ix, iy), stats in self.grid_stats.items():
            if (stats['std_dev_z'] < self.max_cell_std_dev and
                    stats['range_z'] < self.max_cell_range):
                # 将栅格索引映射到矩阵索引
                row = iy - min_iy
                col = ix - min_ix
                is_flat[row, col] = True

        # --- 步骤 2: 找到所有有效的候选中心点 ---
        # 一个点要成为有效中心，它的整个圆形邻域都必须在平坦区域内。
        # 这可以通过一个二维卷积（或更简单的，使用`binary_erosion`）来实现。
        # `binary_erosion` 会将一个点设为True，仅当其结构元素（我们的掩模）覆盖的所有邻域点都为True。

        # 需要反转掩模，因为binary_erosion的结构元素是True表示要检查的区域
        # circular_mask是True表示在圆内，这正好是binary_erosion需要的结构元素
        # 注意：binary_erosion会处理边界，所以边界上的点自动不满足条件
        valid_centers_mask = binary_erosion(is_flat, structure=self.circular_mask)

        # 获取所有有效中心点的矩阵索引
        center_rows, center_cols = np.where(valid_centers_mask)

        if center_rows.size == 0:
            print("未找到合适的着陆区。")
            return None

        # --- 步骤 3: 向量化计算所有有效区域的统计量 ---
        # 将矩阵索引转换回世界栅格索引
        center_ixs = center_cols + min_ix
        center_iys = center_rows + min_iy

        # 计算中心的世界坐标
        center_x_coords = (center_ixs + 0.5) * self.grid_resolution
        center_y_coords = (center_iys + 0.5) * self.grid_resolution

        # 计算到原点的距离
        distances_to_origin = np.sqrt(center_x_coords ** 2 + center_y_coords ** 2)

        # --- 步骤 4: 向量化计算平面度 ---
        # 这是最复杂的一步。我们需要为每个中心点提取其邻域的z值，然后计算标准差。
        # 我们可以使用NumPy的advanced indexing来高效地完成这个操作。

        # 首先，创建一个包含所有单元格平均z值的矩阵，非平坦单元格设为NaN
        avg_z_matrix = np.full((rows, cols), np.nan)
        for (ix, iy), stats in self.grid_stats.items():
            row = iy - min_iy
            col = ix - min_ix
            # 只为平坦单元格设置z值，因为非平坦的不会被考虑
            if is_flat[row, col]:
                avg_z_matrix[row, col] = stats['avg_z']

        # 准备用于提取邻域的索引
        radius = self.cells_in_radius
        y_offsets, x_offsets = np.where(self.circular_mask)
        y_offsets -= radius
        x_offsets -= radius

        # 广播中心点坐标和偏移量，以获取所有邻域的坐标
        # center_rows[:, None] + y_offsets[None, :] 会创建一个矩阵
        # 其中每一行对应一个中心点，每一列对应一个邻域点的行索引
        neighbor_rows = center_rows[:, np.newaxis] + y_offsets[np.newaxis, :]
        neighbor_cols = center_cols[:, np.newaxis] + x_offsets[np.newaxis, :]

        # 使用advanced indexing一次性获取所有邻域的z值
        # 结果是一个二维数组，每一行是一个候选区域的邻域z值列表
        all_neighbor_z_values = avg_z_matrix[neighbor_rows, neighbor_cols]

        # 计算每个候选区域的平面度（标准差）
        # np.nanstd 会自动忽略NaN，但我们这里通过binary_erosion已经保证了邻域内没有NaN
        # 所以用普通的np.std即可，速度更快
        planarity_std_devs = np.std(all_neighbor_z_values, axis=1)

        # 计算每个候选区域的平均高度
        zone_center_zs = np.mean(all_neighbor_z_values, axis=1)

        # --- 步骤 5: 过滤并选择最佳区域 ---
        # 根据平面度标准差进行最终过滤
        final_valid_mask = planarity_std_devs < self.max_zone_planarity_std_dev

        if not np.any(final_valid_mask):
            print("未找到满足最终平面度要求的着陆区。")
            return None

        # 从所有满足条件的候选区中，选择距离原点最近的一个
        # 我们只在有效的候选区中寻找最小距离
        valid_distances = distances_to_origin[final_valid_mask]

        if valid_distances.size == 0:
            return None

        # 找到最小距离的索引（在valid_distances数组中的索引）
        min_dist_idx_in_valid = np.argmin(valid_distances)

        # 找到这个最小距离对应的原始索引（在center_x_coords等数组中的索引）
        # np.where返回一个元组，我们取第一个元素
        original_indices = np.where(final_valid_mask)[0]
        best_original_idx = original_indices[min_dist_idx_in_valid]

        best_zone = {
            'center_x': center_x_coords[best_original_idx],
            'center_y': center_y_coords[best_original_idx],
            'center_z': zone_center_zs[best_original_idx],
            'distance': distances_to_origin[best_original_idx],
            'planarity': planarity_std_devs[best_original_idx]
        }

        return best_zone

    def _find_best_landing_zone0(self):
        """搜索具有最大后续平坦区域的最佳着陆区"""
        if not self.grid_stats:
            return None

        # 1. 创建平坦性矩阵
        all_indices = np.array(list(self.grid_stats.keys()))
        if all_indices.size == 0:
            return None

        min_ix, min_iy = np.min(all_indices, axis=0)
        max_ix, max_iy = np.max(all_indices, axis=0)

        rows = max_iy - min_iy + 1
        cols = max_ix - min_ix + 1
        is_flat = np.zeros((rows, cols), dtype=bool)

        # 创建平均高度矩阵
        avg_z_matrix = np.full((rows, cols), np.nan)

        for (ix, iy), stats in self.grid_stats.items():
            row = iy - min_iy
            col = ix - min_ix
            # 填充平坦性矩阵和高度矩阵
            if (stats['std_dev_z'] < self.max_cell_std_dev and
                    stats['range_z'] < self.max_cell_range):
                is_flat[row, col] = True
                avg_z_matrix[row, col] = stats['avg_z']
            else:
                # 非平坦区域也要填充高度信息（虽然后面可能会用到，但为了完整性）
                # 这里主要关注平坦区域
                pass

        self.flat_area_mask = is_flat  # 保存以便可视化

        # 2. 使用形态学腐蚀找到符合尺寸要求的候选中心
        # valid_centers_mask 保证该点周围的圆形区域都是平坦的
        valid_centers_mask = binary_erosion(is_flat, structure=self.circular_mask)
        center_rows, center_cols = np.where(valid_centers_mask)

        if center_rows.size == 0:
            return None

        # 3. 寻找连通区域
        # 使用 scipy.ndimage.label 找到所有相连的平坦区域
        labeled_array, num_features = label(is_flat)

        # 4. 为每个候选中心点找到其所属的连通区域ID
        candidate_labels = labeled_array[center_rows, center_cols]

        # 5. 计算每个连通区域的大小（包含多少个平坦栅格）
        # 我们需要统计每个label对应的栅格数量
        # np.bincount 非常适合做这件事
        region_sizes = np.bincount(labeled_array.ravel())

        # 此时 region_sizes[0] 是背景（标签0），我们只关心候选区对应的标签
        # 注意：labeled_array中0表示背景，标签从1开始
        # candidate_labels 包含的是每个候选点所在的区域标签

        # 获取每个候选点所在区域的大小
        candidate_region_sizes = region_sizes[candidate_labels]

        # 6. 选择最佳着陆区：周围平坦区域最大的那个
        # 如果区域大小相同，可以按距离原点最近作为次级标准（可选）
        best_idx = np.argmax(candidate_region_sizes)

        # 获取最佳中心点的索引
        best_center_row = center_rows[best_idx]
        best_center_col = center_cols[best_idx]

        # 计算最佳中心的世界坐标
        best_ix = best_center_col + min_ix
        best_iy = best_center_row + min_iy

        center_x = (best_ix + 0.5) * self.grid_resolution
        center_y = (best_iy + 0.5) * self.grid_resolution

        # 提取最佳着陆区的高度（取其圆形邻域的平均高度）
        radius = self.cells_in_radius
        y_offsets, x_offsets = np.where(self.circular_mask)
        y_offsets -= radius
        x_offsets -= radius

        neighbor_rows = best_center_row + y_offsets
        neighbor_cols = best_center_col + x_offsets
        neighbor_z_values = avg_z_matrix[neighbor_rows, neighbor_cols]

        center_z = np.mean(neighbor_z_values)
        planarity_std = np.std(neighbor_z_values)

        # 计算距离原点的距离
        distance = np.sqrt(center_x ** 2 + center_y ** 2)

        # 记录该点所属区域的大小（用于调试）
        area_size = candidate_region_sizes[best_idx]

        best_zone = {
            'center_x': center_x,
            'center_y': center_y,
            'center_z': center_z,
            'distance': distance,
            'planarity': planarity_std,
            # 'area_size': area_size  # 新增：周围平坦区域的面积（栅格数）
        }

        return best_zone