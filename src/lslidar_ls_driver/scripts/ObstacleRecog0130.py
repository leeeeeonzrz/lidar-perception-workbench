from matplotlib import cm, pyplot as plt # 仅用于色彩映射表获取
import open3d as o3d
import numpy as np

import time
from typing import Tuple, List

# 导入工具函数
import utils
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, pyqtSlot,QTimer
from PyQt5.QtCore import QMutex, QMutexLocker
import threading

from multiprocessing import Event

class ObstacleRecog():
    """
    障碍物检测进程类
    """
    def __init__(self):
        #地面分割参数
        self.ground_seg_distance_threshold = 1.6 #1.6
        self.ground_seg_ransac_n = 4
        self.ground_seg_num_iterations = 15

        #聚类参数
        self.nonground_cluster_eps = 4.5 #2
        self.nonground_cluster_min_points_dbscan = 5 #15
        self.nonground_cluster_min_points_for_vis = 20

        self.vehicle_like_min_dim = [1.5, 0.8, 0.8]  # [min_len, min_wid, min_hgt]
        self.vehicle_like_max_dim = [40.0, 40.0, 40.0] # [max_len, max_wid, max_hgt]
        self.vehicle_like_min_points = 50

        self.small_object_max_dim = [1.0, 1.0, 1.5]# [max_len, max_wid, max_hgt]
        self.small_object_min_points = 10

        #点云
        self.pointsAndColors = None
        self.pcd_queue = None

        self.running = True
        self.clustered_informations = []
        self.merged_points = None
        self.merged_colors = None

    def spinOnce0(self, pcd:o3d.geometry.PointCloud, stop_evt: Event = None):

        start_time = time.time()
        self.merged_points = np.empty((0, 3))
        self.merged_colors = np.empty((0, 4))
        self.clustered_informations.clear()
        if pcd.has_points():
            # ground_local_map_W = pcd.voxel_down_sample(voxel_size=0.9)
            ground_local_map_W, nonground_local_map_W = self.segment_ground_ransac(pcd,
                                                                                   self.ground_seg_distance_threshold,
                                                                                   self.ground_seg_ransac_n,
                                                                                   self.ground_seg_num_iterations)
            end_time2 = time.time()
            elapsed_time2 = end_time2 - start_time
            # print(f"分割运行时间: {elapsed_time2:.4f} 秒")
            # print(
            #     f"  局部地图分割 (W): 地面 {len(ground_local_map_W.points)}, 非地面 {len(nonground_local_map_W.points)}")

            ground_points_np = np.asarray(ground_local_map_W.points)
            ground_colors_np = np.array([[0.5, 0.5, 0.5, 1.0]] * len(ground_local_map_W.points))

            no_ground_points_np = None
            no_ground_colors_np = None

            if len(nonground_local_map_W.points):
                no_ground_points_np, no_ground_colors_np = self.cluster_nonground_points(
                    nonground_local_map_W,
                    self.nonground_cluster_eps,
                    self.nonground_cluster_min_points_dbscan,
                    self.nonground_cluster_min_points_for_vis
                )

            end_time3 = time.time()
            elapsed_time3 = end_time3 - end_time2
            # print(f"聚类运行时间: {elapsed_time3:.4f} 秒")
            if no_ground_points_np is not None and len(no_ground_points_np) > 0:
                self.merged_points = np.vstack([ground_points_np, no_ground_points_np])
                self.merged_colors = np.vstack([ground_colors_np, no_ground_colors_np])
            else:
                self.merged_points = ground_points_np
                self.merged_colors = ground_colors_np

        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time_ms = elapsed_time * 1000
        # print(f"代码运行时间: {elapsed_time:.4f} 秒")
        target_duration_ms = 95.0  # 目标运行时间（毫秒）
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
        # print(f"总运行时间: {final_elapsed:.2f} ms")

    def spinOnce(self, pcd:o3d.geometry.PointCloud, stop_evt: Event = None):

        start_time = time.time()
        self.merged_points = np.empty((0, 3))
        self.merged_colors = np.empty((0, 4))
        self.clustered_informations.clear()
        if pcd.has_points():

            points_np = np.asarray(pcd.points)
            # --- 核心修改：分流逻辑 ---
            # 1. 计算点云中心 (x, y, z)
            center = np.mean(points_np, axis=0)
            # 2. 计算中心点到原点 (0,0,0) 的 3D 欧几里得距离
            distance = np.linalg.norm(center)
            #distance = np.abs(center[1])

            if distance > 30.0:

                down_sampled_pcd = pcd.voxel_down_sample(voxel_size=0.5)

                # 2. 旋转点云
                # points_np = np.asarray(down_sampled_pcd.points)
                # points_rotated = utils.rotate_points_around_x_axis(points_np, -20.0)
                # down_sampled_pcd.points = o3d.utility.Vector3dVector(points_rotated)

                # ground_local_map_W = pcd.voxel_down_sample(voxel_size=0.9)
                ground_local_map_W, nonground_local_map_W = self.segment_ground_ransac(down_sampled_pcd,
                                                                                       self.ground_seg_distance_threshold,
                                                                                       self.ground_seg_ransac_n,
                                                                                       self.ground_seg_num_iterations)
                end_time2 = time.time()
                elapsed_time2 = end_time2 - start_time
                # print(f"分割运行时间: {elapsed_time2:.4f} 秒")
                # print(
                #     f"  局部地图分割 (W): 地面 {len(ground_local_map_W.points)}, 非地面 {len(nonground_local_map_W.points)}")

                ground_points_np = np.asarray(ground_local_map_W.points)
                ground_colors_np = np.array([[0.5, 0.5, 0.5, 1.0]] * len(ground_local_map_W.points))

                no_ground_points_np = None
                no_ground_colors_np = None

                if len(nonground_local_map_W.points):
                    no_ground_points_np, no_ground_colors_np = self.cluster_nonground_points(
                        nonground_local_map_W,
                        self.nonground_cluster_eps,
                        self.nonground_cluster_min_points_dbscan,
                        self.nonground_cluster_min_points_for_vis
                    )

                end_time3 = time.time()
                elapsed_time3 = end_time3 - end_time2
                # print(f"聚类运行时间: {elapsed_time3:.4f} 秒")
                if no_ground_points_np is not None and len(no_ground_points_np) > 0:
                    self.merged_points = np.vstack([ground_points_np, no_ground_points_np])
                    self.merged_colors = np.vstack([ground_colors_np, no_ground_colors_np])
                else:
                    self.merged_points = ground_points_np
                    self.merged_colors = ground_colors_np

            else:
                # >>> 场景 B：Y轴距离 <= 30m -> 体素降采样 + 直接聚类 <<<
                interior_points, interior_colors = self.process_interior(pcd)
                self.merged_points = interior_points
                self.merged_colors = interior_colors

        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time_ms = elapsed_time * 1000
        # print(f"代码运行时间: {elapsed_time:.4f} 秒")
        target_duration_ms = 95.0  # 目标运行时间（毫秒）
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
        # print(f"总运行时间: {final_elapsed:.2f} ms")


    def process_interior(self, pcd: o3d.geometry.PointCloud):
        """
        针对室内场景的处理函数：
        1. 体素降采样
        2. 直接进行聚类 (不做地面分割，保留未聚类点为白色)
        """
        points = np.asarray(pcd.points)
        pcd.points = o3d.utility.Vector3dVector(points)

        # 体素降采样
        down_pcd = pcd.voxel_down_sample(voxel_size=0.08)

        if down_pcd.is_empty():
            return np.empty((0, 3)), np.empty((0, 4))

        # 直接聚类
        # 注意：现在返回的 clustered_points 包含了所有点（聚类成功的彩色 + 噪声的白色）
        clustered_points, clustered_colors = self.cluster_indoor_points(
            down_pcd,
            0.15,  # eps
            15,  # min_points_dbscan
            30  # min_cluster_size_for_vis
        )

        return clustered_points, clustered_colors

    def segment_ground_ransac(self, pcd: o3d.geometry.PointCloud,
                              distance_threshold: float,
                              ransac_n: int,
                              num_iterations: int) -> Tuple:
        """
        使用RANSAC分割地面点云。
        返回: (ground_pcd, nonground_pcd)
        """

        if pcd is None or pcd.is_empty() or len(pcd.points) < ransac_n:
            # rospy.logdebug_throttle(5.0, "segment_ground_ransac: 输入点云为空或点数不足。")

            return o3d.geometry.PointCloud(), o3d.geometry.PointCloud(pcd) if pcd else o3d.geometry.PointCloud()


        try:
            # segment_plane 会找到最大的平面
            plane_model, inlier_indices = pcd.segment_plane(distance_threshold=distance_threshold,
                                                            ransac_n=ransac_n,
                                                            num_iterations=num_iterations)
            # 可选: 检查平面法向量是否大致朝上 (例如 Z 分量 > 0.7)
            # normal_z = plane_model[2]
            # if abs(normal_z) < 0.7: # 假设世界Z轴向上
            #     rospy.logdebug_throttle(5.0, f"RANSAC平面法向量Z分量 {normal_z:.2f} 过小，可能不是水平地面。将所有点视为非地面。")
            #     return o3d.geometry.PointCloud(), o3d.geometry.PointCloud(pcd)

        except RuntimeError as e:
            print(f"地面分割RANSAC错误: {e}. 将所有点视为非地面。")
            return o3d.geometry.PointCloud(), o3d.geometry.PointCloud(pcd)

        ground_pcd = pcd.select_by_index(inlier_indices)
        nonground_pcd = pcd.select_by_index(inlier_indices, invert=True)

        return ground_pcd, nonground_pcd



    def cluster_nonground_points0(self, nonground_pcd: o3d.geometry.PointCloud,
                                           eps: float,
                                           min_points_dbscan: int,
                                           min_cluster_size_for_vis: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        优化版：对非地面点云进行欧式聚类 (DBSCAN) 并为簇着色。
        优化点：
        1. 合并 np.unique 调用，减少重复计算。
        2. 用 np.isin 替代列表推导式生成 valid_mask。
        3. 跳过不必要的排序（如果后续处理不要求连续内存）。
        4. 预分配颜色数组。
        """
        if nonground_pcd is None or nonground_pcd.is_empty() or len(nonground_pcd.points) < min_points_dbscan:
            return np.empty((0, 3)), np.empty((0, 4))

        try:
            labels = np.array(nonground_pcd.cluster_dbscan(eps=eps, min_points=min_points_dbscan, print_progress=False))
        except RuntimeError as e:
            print(f"DBSCAN聚类错误: {e}. 返回空聚类点云。")
            return np.empty((0, 3)), np.empty((0, 4))

        max_label = labels.max()
        # self.clustered_informations.clear()

        if max_label < 0:
            return np.empty((0, 3)), np.empty((0, 4))

        points_np = np.asarray(nonground_pcd.points)

        # 优化1: 合并 np.unique 调用
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_cluster_labels = unique_labels[(unique_labels != -1) & (counts >= min_cluster_size_for_vis)]

        if len(valid_cluster_labels) == 0:
            return np.empty((0, 3)), np.empty((0, 4))

        # 优化2: 用 np.isin 替代列表推导式
        valid_mask = np.isin(labels, valid_cluster_labels)
        valid_points = points_np[valid_mask]
        valid_labels = labels[valid_mask]

        # 优化3: 跳过排序（如果后续处理不要求连续内存）
        # 直接通过 valid_labels 分组计算边界框
        cluster_labels = np.unique(valid_labels)
        cluster_indices = [np.where(valid_labels == label)[0] for label in cluster_labels]

        # 向量化计算边界框
        min_coords = np.array([valid_points[idx].min(axis=0) for idx in cluster_indices])
        max_coords = np.array([valid_points[idx].max(axis=0) for idx in cluster_indices])
        centers = (min_coords + max_coords) / 2.0
        extents = max_coords - min_coords

        self.clustered_informations = [
            {'center': centers[i], 'extent': extents[i]}
            for i in range(len(cluster_labels))
        ]

        # 优化4: 预分配颜色数组
        num_valid_clusters = len(cluster_labels)
        colors = np.zeros((len(valid_points), 4))
        for i, label in enumerate(cluster_labels):
            mask = (valid_labels == label)
            color = plt.cm.hsv(i / max(1, num_valid_clusters - 1))[:3]
            colors[mask] = [*color, 1.0]

        return valid_points, colors

    def cluster_nonground_points1(self, nonground_pcd: o3d.geometry.PointCloud,
                                           eps: float,
                                           min_points_dbscan: int,
                                           min_cluster_size_for_vis: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        对非地面点云进行欧式聚类 (DBSCAN) 并为簇着色。
        优化点：
        1. 合并 np.unique 调用，减少重复计算。
        2. 用 np.isin 替代列表推导式生成 valid_mask。
        3. 跳过不必要的排序（如果后续处理不要求连续内存）。
        4. 预分配颜色数组。
        """
        if nonground_pcd is None or nonground_pcd.is_empty() or len(nonground_pcd.points) < min_points_dbscan:
            return np.empty((0, 3)), np.empty((0, 4))

        try:
            labels = np.array(nonground_pcd.cluster_dbscan(eps=eps, min_points=min_points_dbscan, print_progress=False))
        except RuntimeError as e:
            print(f"DBSCAN聚类错误: {e}. 返回空聚类点云。")
            return np.empty((0, 3)), np.empty((0, 4))

        max_label = labels.max()
        # self.clustered_informations.clear()

        if max_label < 0:
            return np.empty((0, 3)), np.empty((0, 4))

        points_np = np.asarray(nonground_pcd.points)

        # 优化1: 合并 np.unique 调用
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_cluster_labels = unique_labels[(unique_labels != -1) & (counts >= min_cluster_size_for_vis)]

        if len(valid_cluster_labels) == 0:
            return np.empty((0, 3)), np.empty((0, 4))

        # 优化2: 用 np.isin 替代列表推导式
        valid_mask = np.isin(labels, valid_cluster_labels)
        valid_points = points_np[valid_mask]
        valid_labels = labels[valid_mask]

        # --- 核心优化部分 ---
        # 1. 对标签进行排序，这样相同的标签就会聚集在一起
        sort_indices = np.argsort(valid_labels)
        sorted_valid_labels = valid_labels[sort_indices]
        sorted_valid_points = valid_points[sort_indices]

        # 2. 使用 np.unique 的 return_index=True 来找到每个簇的起始位置
        #    这比 np.where 快得多
        cluster_labels, start_indices = np.unique(sorted_valid_labels, return_index=True)

        # 3. 计算每个簇的结束位置
        end_indices = np.concatenate([start_indices[1:], [len(sorted_valid_labels)]])

        # 4. 现在可以高效地遍历每个簇了
        for i, label in enumerate(cluster_labels):
            start_idx = start_indices[i]
            end_idx = end_indices[i]

            # 直接通过切片获取簇的点，这比用索引数组快
            cluster_points_array = sorted_valid_points[start_idx:end_idx]


            obb_info = self.compute_z_aligned_obb(cluster_points_array)

            self.clustered_informations.append({
                'center': obb_info['center'],
                'extent': obb_info['extent'],
                'rotation': obb_info['rotation']
            })

        # 优化4: 预分配颜色数组
        num_valid_clusters = len(cluster_labels)
        colors = np.zeros((len(valid_points), 4))
        for i, label in enumerate(cluster_labels):
            mask = (valid_labels == label)
            color = plt.cm.hsv(i / max(1, num_valid_clusters - 1))[:3]
            colors[mask] = [*color, 1.0]

        return valid_points, colors

    def cluster_nonground_points(self, nonground_pcd: o3d.geometry.PointCloud,
                                  eps: float,
                                  min_points_dbscan: int,
                                  min_cluster_size_for_vis: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        对非地面点云进行欧式聚类 (DBSCAN) 并为簇着色。

        """
        if nonground_pcd is None or nonground_pcd.is_empty() or len(nonground_pcd.points) < min_points_dbscan:
            return np.empty((0, 3)), np.empty((0, 4))

        try:
            labels = np.array(nonground_pcd.cluster_dbscan(eps=eps, min_points=min_points_dbscan, print_progress=False))
        except RuntimeError as e:
            print(f"DBSCAN聚类错误: {e}. 返回空聚类点云。")
            return np.empty((0, 3)), np.empty((0, 4))

        max_label = labels.max()
        # self.clustered_informations.clear()

        if max_label < 0:
            return np.empty((0, 3)), np.empty((0, 4))

        points_np = np.asarray(nonground_pcd.points)

        # 优化1: 合并 np.unique 调用
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_cluster_labels = unique_labels[(unique_labels != -1) & (counts >= min_cluster_size_for_vis)]

        if len(valid_cluster_labels) == 0:
            return np.empty((0, 3)), np.empty((0, 4))

        # 优化2: 用 np.isin 替代列表推导式
        valid_mask = np.isin(labels, valid_cluster_labels)
        valid_points = points_np[valid_mask]
        valid_labels = labels[valid_mask]

        # --- 核心优化部分 ---
        # 1. 对标签进行排序，这样相同的标签就会聚集在一起
        sort_indices = np.argsort(valid_labels)
        sorted_valid_labels = valid_labels[sort_indices]
        sorted_valid_points = valid_points[sort_indices]

        # 2. 使用 np.unique 的 return_index=True 来找到每个簇的起始位置
        #    这比 np.where 快得多
        cluster_labels, start_indices = np.unique(sorted_valid_labels, return_index=True)

        # 3. 计算每个簇的结束位置
        end_indices = np.concatenate([start_indices[1:], [len(sorted_valid_labels)]])

        colors = np.zeros((len(valid_points), 4))
        # 4. 现在可以高效地遍历每个簇了
        for i, label in enumerate(cluster_labels):
            start_idx = start_indices[i]
            end_idx = end_indices[i]

            # 直接通过切片获取簇的点，这比用索引数组快
            cluster_points_array = sorted_valid_points[start_idx:end_idx]

            obb_info = self.compute_z_aligned_obb(cluster_points_array)

            self.clustered_informations.append({
                'center': obb_info['center'],
                'extent': obb_info['extent'],
                'rotation': obb_info['rotation']
            })
            # --- 新增：语义识别和着色 ---
            category, features = self.classify_cluster_features(cluster_points_array)

            # 根据类别分配颜色
            if category == "building":
                colors[valid_labels == label] = [0.0, 0.0, 1.0, 1.0]  # 蓝色
            elif category == "vegetation":
                colors[valid_labels == label] = [0.0, 1.0, 0.0, 1.0]  # 绿色
            elif category == "pole":
                colors[valid_labels == label] = [1.0, 0.0, 0.0, 1.0]  # 红色
            else:
                colors[valid_labels == label] = [0.5, 0.5, 0.5, 1.0]  # 灰色

        return valid_points, colors

    def cluster_indoor_points(self, nonground_pcd: o3d.geometry.PointCloud,
                                 eps: float,
                                 min_points_dbscan: int,
                                 min_cluster_size_for_vis: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        对非地面点云进行欧式聚类 (DBSCAN) 并为簇着色。
        修改点：
        1. 返回聚类成功的点（彩色）。
        2. 返回未聚类成功的噪声点（白色）。
        """
        if nonground_pcd is None or nonground_pcd.is_empty() or len(nonground_pcd.points) < min_points_dbscan:
            return np.empty((0, 3)), np.empty((0, 4))

        try:
            labels = np.array(nonground_pcd.cluster_dbscan(eps=eps, min_points=min_points_dbscan, print_progress=False))
        except RuntimeError as e:
            print(f"DBSCAN聚类错误: {e}. 返回空聚类点云。")
            return np.empty((0, 3)), np.empty((0, 4))

        points_np = np.asarray(nonground_pcd.points)

        # --- 新增逻辑：分离有效簇和噪声 ---
        # 1. 统计每个标签的点数
        unique_labels, counts = np.unique(labels, return_counts=True)

        # 2. 找出符合大小要求的“有效簇”标签 (排除 -1)
        valid_cluster_labels = unique_labels[(unique_labels != -1) & (counts >= min_cluster_size_for_vis)]

        # --- 处理有效簇 (彩色) ---
        cluster_indices = np.isin(labels, valid_cluster_labels)
        clustered_points = points_np[cluster_indices]
        clustered_labels = labels[cluster_indices]

        # 准备颜色数组
        colors = np.zeros((len(points_np), 4))  # 注意：这里大小是总点数，因为要包含噪声点

        if len(clustered_points) > 0:
            # 排序以便遍历
            sort_indices = np.argsort(clustered_labels)
            sorted_cluster_labels = clustered_labels[sort_indices]
            sorted_cluster_points = clustered_points[sort_indices]

            # 获取每个簇的起止位置
            cluster_labels_unique, start_indices = np.unique(sorted_cluster_labels, return_index=True)
            end_indices = np.concatenate([start_indices[1:], [len(sorted_cluster_labels)]])

            # 遍历簇计算 OBB
            for i, label in enumerate(cluster_labels_unique):
                start_idx = start_indices[i]
                end_idx = end_indices[i]
                cluster_points_array = sorted_cluster_points[start_idx:end_idx]

                # --- 计算 OBB ---
                obb_info = self.compute_z_aligned_obb(cluster_points_array)
                self.clustered_informations.append({
                    'center': obb_info['center'],
                    'extent': obb_info['extent'],
                    'rotation': obb_info['rotation']
                })

                # --- 簇着色 (彩色) ---
                # 注意：这里的 mask 是针对 clustered_points 的，我们需要映射回原始的 points_np
                # 使用 cluster_indices 这个布尔数组来映射
                mask_for_full = cluster_indices.copy()
                # 因为 clustered_points 是 points_np[cluster_indices] 的子集，
                # 但 clustered_points 内部排序了，所以直接用 mask_for_full 不好对应颜色。
                # 简单做法：遍历 valid_cluster_labels 赋值
                pass

            # --- 修正后的着色逻辑 (更稳健) ---
            # 先把所有点初始化为白色 (噪声色)
            colors[:] = [0.5, 0.5, 0.5, 0.5]

            # 只对有效簇的点进行着色
            num_valid_clusters = len(cluster_labels_unique)
            for i, label in enumerate(cluster_labels_unique):
                # 找到该簇在原始 labels 中的位置
                mask = (labels == label)
                color = plt.cm.hsv(i / max(1, num_valid_clusters - 1))[:3]
                colors[mask] = [*color, 1.0]  # 覆盖白色

            # 没有被 valid_cluster_labels 覆盖的点（包括噪声和过小的簇）保持白色
            # 其中标签为 -1 的是噪声，标签存在但点数不足的是小物体（也被视为白色）

        else:
            # 没有任何有效簇，全部为白色
            colors[:] = [0.5, 0.5, 0.5, 0.5]

        return points_np, colors

    def compute_bounding_box(self, points_np):
        """计算点云的三维包围框"""
        if points_np.size == 0:
            return None, None, None

        # 计算最小和最大坐标值
        min_coords = np.min(points_np, axis=0)
        max_coords = np.max(points_np, axis=0)

        # 计算中心点
        center = (min_coords + max_coords) / 2.0

        # 计算尺寸 (长宽高)
        dimensions = max_coords - min_coords

        return (center, dimensions)

    def compute_z_aligned_obb(self, points):
        """
        计算一个Z轴对齐的最小包围框。
        包围框在XY平面内自适应旋转，但在Z轴方向保持垂直。

        Args:
            points (np.ndarray): Nx3 的点云数组

        Returns:
            dict: 包含 'center', 'extent', 'rotation' 的字典
        """
        if len(points) < 3:
            # 点太少，返回一个点大小的包围框
            center = points.mean(axis=0)
            return {
                'center': center,
                'extent': np.zeros(3),
                'rotation': np.eye(3)
            }

        # 1. 投影到XY平面并计算2D PCA
        points_xy = points[:, :2]
        mean_xy = points_xy.mean(axis=0)
        cov_xy = np.cov((points_xy - mean_xy).T)

        # 计算特征向量和特征值
        eigvals, eigvecs = np.linalg.eigh(cov_xy)

        # 按特征值从大到小排序，得到主方向
        sort_indices = eigvals.argsort()[::-1]
        principal_axis_2d = eigvecs[:, sort_indices[0]]

        # 2. 构建3D旋转矩阵
        z_axis = np.array([0, 0, 1])

        # 计算出新的X轴（主方向）和Y轴
        new_x_axis = principal_axis_2d
        # --- 核心修改：将2D向量扩展为3D ---
        new_x_axis_3d = np.append(new_x_axis, 0)

        # 确保Y轴与X轴和Z轴都垂直
        # 注意：这里也要用3D的x轴来计算
        new_y_axis = np.cross(z_axis, new_x_axis_3d)

        # 组合成旋转矩阵
        # --- 这里也要用3D的向量 ---
        rotation_matrix = np.vstack([new_x_axis_3d, new_y_axis, z_axis]).T
        # 3. 旋转所有点并计算AABB
        # (points - center) @ R.T 等价于 R_inv @ (points - center)
        # 这里我们直接旋转点云
        rotated_points = points @ rotation_matrix

        min_bound = rotated_points.min(axis=0)
        max_bound = rotated_points.max(axis=0)

        # 4. 计算最终的中心和尺寸
        extent = max_bound - min_bound
        center_rotated = (min_bound + max_bound) / 2.0

        # 将中心点旋转回世界坐标系
        center_world = center_rotated @ rotation_matrix.T

        return {
            'center': center_world,
            'extent': extent,
            'rotation': rotation_matrix
        }


    def extract_cluster_features(self, points: np.ndarray) -> dict:
        """
        提取聚类点云的核心特征
        """
        if len(points) < 3:
            return None

        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(points)

        # 计算包围盒
        obb = temp_pcd.get_oriented_bounding_box()
        size_obb = obb.extent

        volume = np.prod(size_obb)
        height = size_obb[2]

        # 俯视长宽比
        if min(size_obb[:2]) != 0:
            aspect_ratio_2d = max(size_obb[:2]) / min(size_obb[:2])
        else:
            aspect_ratio_2d = 0

        # PCA形状特征（2D）
        points_2d = points[:, :2]
        mean_2d = points_2d.mean(axis=0)
        cov_2d = np.cov((points_2d - mean_2d).T)

        eig_vals_2d, _ = np.linalg.eig(cov_2d)
        eig_vals_2d = np.sort(np.real(eig_vals_2d))[::-1]

        if eig_vals_2d[0] != 0:
            linearity_2d = (eig_vals_2d[0] - eig_vals_2d[1]) / eig_vals_2d[0]
        else:
            linearity_2d = 0

        # 统计特征
        point_num = len(points)
        density = point_num / volume if volume != 0 else 0

        # 凸性特征
        try:
            convex_hull = temp_pcd.compute_convex_hull()
            hull_vol = convex_hull[0].get_volume() if hasattr(convex_hull[0], 'get_volume') else 0
            convexity = hull_vol / volume if volume != 0 else 0
        except:
            convexity = 0

        return {
            "height": height,
            "volume": volume,
            "aspect_ratio_2d": aspect_ratio_2d,
            "linearity_2d": linearity_2d,
            "convexity": convexity,
            "density": density,
            "point_num": point_num
        }


    def classify_cluster_features(self, points: np.ndarray) -> Tuple[str, dict]:
        """
        对聚类点云进行分类，返回类别和特征信息

        Returns:
            tuple: (类别名称, 特征字典)
        """
        # 提取特征
        features = self.extract_cluster_features(points)

        if features is None:
            return "unknown", {}

        h = features["height"]
        v = features["volume"]
        ar = features["aspect_ratio_2d"]
        lin = features["linearity_2d"]
        con = features["convexity"]
        den = features["density"]
        print(f": H={h:.2f}, V={v:.2f}, 凸性={con:.2f}, 密度={den:.2f}, 长宽比={ar:.2f}")

        # 分类规则
        # 1. 塔杆（优先判断）
        # 特征：细长（线性度极高），体积非常小，高度较高
        # 排除树丛：树丛体积通常较大
        if 3 < h < 60 and v < 30 and lin > 0.85 and ar > 6:
            return "pole", features

        if (
                h > 1.0 and  # 高度降低：原来5 -> 现在2.5 (覆盖小平房)
                v > 10 and  # 体积降低：原来20 -> 现在10 (覆盖小棚子)
                #con > 0.75 and  # 凸性提高：原来0.7 -> 现在0.75 (必须是规则形状)
                #den > 0.4 and  # 密度保持：> 0.4 (实体物体)
                ar < 6  # 长宽比限制：< 6 (排除细长树丛)
        ):
            return "building", features

        # 2. 植被（重点优化：包含单棵树和一排树）
        # 特征：
        # - 高度：2m ~ 25m（树丛也不会无限高）
        # - 凸性：< 0.65（核心指标：树丛/树林形状不规则，建筑通常 > 0.7）
        # - 密度：< 0.5（核心指标：植被稀疏，建筑/车辆致密）
        # - 体积和长宽比：不做严格上限，允许很大（覆盖树丛）
        elif 2 < h < 25 and con < 0.65 and den < 0.5:
            # 可以进一步打印看看是不是树丛
            if v > 100 or ar > 4:
                # print(f"检测到植被簇: 体积={v:.1f}, 长宽比={ar:.1f} (可能是树丛)")
                pass
            return "vegetation", features

        # 未知目标
        else:
            return "unknown", features