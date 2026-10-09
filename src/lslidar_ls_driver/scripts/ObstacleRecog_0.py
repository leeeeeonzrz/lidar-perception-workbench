from matplotlib import cm, pyplot as plt # 仅用于色彩映射表获取
import open3d as o3d
import numpy as np

import time
import logging
from typing import Tuple, List

# 导入工具函数
# import utils
# from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, pyqtSlot,QTimer
# from PyQt5.QtCore import QMutex, QMutexLocker
# import threading

from multiprocessing import Event
'''
功能：障碍物检测


输入：点云
输出：障碍物点云

流程：
1. 地面分割
2. 非地面点云聚类
3. 障碍物点云识别

算法：
1. 地面分割：使用RANSAC算法分割地面点云
2. 非地面点云聚类：使用DBSCAN算法聚类非地面点云
3. 障碍物点云识别：使用OBB算法识别障碍物点云

对外接口：
1. spinOnce(pcd:o3d.geometry.PointCloud, stop_evt: Event = None)：处理一帧点云数据
2. process_interior(pcd: o3d.geometry.PointCloud)：处理室内点云数据
3. segment_ground_ransac(pcd: o3d.geometry.PointCloud, distance_threshold: float, ransac_n: int, num_iterations: int) -> Tuple：使用RANSAC算法分割地面点云
4. cluster_nonground_points(nonground_pcd: o3d.geometry.PointCloud, eps: float, min_points_dbscan: int, min_cluster_size_for_vis: int) -> Tuple[np.ndarray, np.ndarray]：使用DBSCAN算法聚类非地面点云
5. cluster_indoor_points(nonground_pcd: o3d.geometry.PointCloud, eps: float, min_points_dbscan: int, min_cluster_size_for_vis: int) -> Tuple[np.ndarray, np.ndarray]：使用DBSCAN算法聚类室内点云
6. compute_z_aligned_obb(points)：计算Z轴对齐的最小包围框
7. extract_cluster_features(points: np.ndarray) -> dict：提取聚类点云的核心特征
8. classify_cluster_features(points: np.ndarray) -> Tuple[str, dict]：分类聚类点云的核心特征

'''
class ObstacleRecog():
    """
    障碍物检测进程类
    """
    def __init__(self):
        # 初始化日志记录器
        # 在多进程环境下，需要确保日志配置存在
        self.logger = logging.getLogger(__name__)
        
        # 如果logger还没有handler，配置日志输出到文件和控制台
        if not self.logger.handlers:
            import os
            log_file = os.path.join(os.path.dirname(__file__), 'obstacle_recog.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            # # 控制台输出
            # console_handler = logging.StreamHandler()
            # console_handler.setFormatter(formatter)
            # self.logger.addHandler(console_handler)
            
            # 文件输出
            try:
                file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception:
                pass  # 如果无法创建文件，只输出到控制台
            
            self.logger.setLevel(logging.INFO)
            # 防止日志向上传播到root logger（避免重复输出）
            self.logger.propagate = False
        
        #地面分割参数
        self.ground_seg_distance_threshold = 1.8 #1.6
        self.ground_seg_ransac_n = 4
        self.ground_seg_num_iterations = 15

        #聚类参数

        self.nonground_cluster_eps = 4.5 #2   4.5  3.5
        self.nonground_cluster_min_points_dbscan = 15 #15   5
        self.nonground_cluster_min_points_for_vis = 20 #    20

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

    def spinOnce(self, pcd:o3d.geometry.PointCloud, stop_evt: Event = None): #lq 修改了输入输出

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

            self.point_min_z = np.min(points_np[:, 2])
            center_z = -center[2]

            if center_z > 3:
            # if distance > 30.0:
                
                down_sampled_pcd = pcd.voxel_down_sample(voxel_size=0.5) # 0.5

                ground_local_map_W, nonground_local_map_W = self.segment_ground_ransac(down_sampled_pcd,
                                                                                       self.ground_seg_distance_threshold,
                                                                                       self.ground_seg_ransac_n,
                                                                                       self.ground_seg_num_iterations)

                ground_local_map_W, nonground_local_map_W = self.segment_ground_ransac(down_sampled_pcd,
                                                                                       1.6,
                                                                                       4,
                                                                                       15)
                
                end_time2 = time.time()
                elapsed_time2 = end_time2 - start_time
                # print(f"分割运行时间: {elapsed_time2:.4f} 秒")
                # print(
                #     f"  局部地图分割 (W): 地面 {len(ground_local_map_W.points)}, 非地面 {len(nonground_local_map_W.points)}")

                ground_points_np = np.asarray(ground_local_map_W.points)
                ground_colors_np = np.array([[0.5, 0.5, 0.5, 1.0]] * len(ground_local_map_W.points))

                no_ground_points_np = None
                no_ground_colors_np = None

                # if len(nonground_local_map_W.points):
                #     no_ground_points_np, no_ground_colors_np = self.cluster_nonground_points(
                #         nonground_local_map_W,
                #         self.nonground_cluster_eps,
                #         self.nonground_cluster_min_points_dbscan,
                #         self.nonground_cluster_min_points_for_vis
                #     )
                if len(nonground_local_map_W.points):
                    no_ground_points_np, no_ground_colors_np = self.cluster_nonground_points(
                        nonground_local_map_W,
                        4.5,
                        5,
                        20
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

    def spinOnce_no_semantic(self, pcd: o3d.geometry.PointCloud, stop_evt: Event = None):  # lq 修改了输入输出

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
            # distance = np.abs(center[1])

            center_z = -center[2]

            if center_z > 3:
            # if distance > 30.0:

                down_sampled_pcd = pcd.voxel_down_sample(voxel_size=0.5)  # 0.5

                # 2. 旋转点云
                # points_np = np.asarray(down_sampled_pcd.points)
                # points_rotated = utils.rotate_points_around_x_axis(points_np, -20.0)
                # down_sampled_pcd.points = o3d.utility.Vector3dVector(points_rotated)

                # ground_local_map_W = pcd.voxel_down_sample(voxel_size=0.9)
                ground_local_map_W, nonground_local_map_W = self.segment_ground_ransac(down_sampled_pcd,
                                                                                       1.6,
                                                                                       4,
                                                                                       15)

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
                    no_ground_points_np, no_ground_colors_np = self.cluster_nonground_points1(
                        nonground_local_map_W,
                        4.5,
                        5,
                        20
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

    def cluster_nonground_points2(self, nonground_pcd: o3d.geometry.PointCloud,
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

        colors = np.full((len(valid_points), 4), [0.5, 0.5, 0.5, 1.0]) #默认颜色为灰色
        print("---------------------------")
        # 4. 现在可以高效地遍历每个簇了
        for i, label in enumerate(cluster_labels):
            start_idx = start_indices[i]
            end_idx = end_indices[i]

            # 直接通过切片获取簇的点，这比用索引数组快
            cluster_points_array = sorted_valid_points[start_idx:end_idx]

            # --- 新增：语义识别和着色 ---
           
            category, features = self.classify_cluster_features(cluster_points_array)

            # 判断高度，如果小于2米，则跳过后续逻辑
            if features['height'] < 2 or (features['volume'] < 30 and features['z_range'] < 4):
                print("滤除了一些聚类")
                continue  # 跳过当前循环，不计算OBB，不添加颜色，不入库

            obb_info = self.compute_z_aligned_obb(cluster_points_array)

            self.clustered_informations.append({
                'center': obb_info['center'],
                'extent': obb_info['extent'],
                'rotation': obb_info['rotation']
            })

            self.logger.info("--------------------------------")
            self.logger.info(f"类别: {category}")
            self.logger.info(f"高度: {features['height']:.3f}, 体积: {features['volume']:.3f}, 长宽比: {features['aspect_ratio_2d']:.3f}, 线性度: {features['linearity_2d']:.3f}, 凸性: {features['convexity']:.3f}, 密度: {features['density']:.3f}")
            self.logger.info(f"z均值: {features['z_mean']:.3f}, z标准差: {features['z_std']:.3f}, z范围: {features['z_range']:.3f},  z变异系数: {features['z_cv']:.3f}")
            self.logger.info("--------------------------------")

            # # 根据类别分配颜色
            if category == "building":
                colors[valid_labels == label] = [0.0, 0.0, 1.0, 1.0]  # 蓝色
            elif category == "vegetation":
                colors[valid_labels == label] = [0.0, 1.0, 0.0, 1.0]  # 绿色
            elif category == "pole":
                colors[valid_labels == label] = [1.0, 0.0, 0.0, 1.0]  # 红色
            else:
                colors[valid_labels == label] = [1.0, 1.0, 0.0, 1.0]  # 黄色
        return valid_points, colors
    
    def cluster_nonground_points(self, nonground_pcd: o3d.geometry.PointCloud,
                              eps: float,
                              min_points_dbscan: int,
                              min_cluster_size_for_vis: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        对非地面点云进行欧式聚类 (DBSCAN) 并为簇着色。
        返回的点云坐标与输入完全一致，仅颜色不同。
        未被聚类或被逻辑跳过的点均为白色。
        """
        if nonground_pcd is None or nonground_pcd.is_empty() or len(nonground_pcd.points) < min_points_dbscan:
            # 如果输入为空，返回空，颜色为空
            return np.empty((0, 3)), np.empty((0, 4))
        try:
            labels = np.array(nonground_pcd.cluster_dbscan(eps=eps, min_points=min_points_dbscan, print_progress=False))
        except RuntimeError as e:
            print(f"DBSCAN聚类错误: {e}. 返回原始点云（全白）。")
            points_np = np.asarray(nonground_pcd.points)
            colors = np.full((len(points_np), 4), [0.5, 0.5, 0.5, 0.5]) # 出错也返回全白
            return points_np, colors
        # 1. 初始化所有点为白色 (未被聚类的点保持白色)
        # 注意：这里直接用输入点云的数量来初始化，确保一一对应
        points_np = np.asarray(nonground_pcd.points)
        colors = np.full((len(points_np), 4), [0.5, 0.5, 0.5, 0.5])
        max_label = labels.max()
        self.clustered_informations.clear()
        if max_label < 0:
            # 没有任何聚类，直接返回全白点云
            return points_np, colors
        # 优化1: 合并 np.unique 调用，统计每个标签的点数
        unique_labels, counts = np.unique(labels, return_counts=True)

        # 筛选出有效的聚类标签（非噪声 且 点数足够）
        # 注意：这里的筛选只是为了确定哪些簇需要进入详细的特征计算循环
        valid_cluster_labels = unique_labels[(unique_labels != -1) & (counts >= min_cluster_size_for_vis)]
        if len(valid_cluster_labels) == 0:
            # 没有满足大小条件的簇，返回全白点云
            return points_np, colors

        is_valid_cluster = np.isin(labels, valid_cluster_labels)

        # 提取有效点进行加速处理
        valid_points = points_np[is_valid_cluster]
        valid_labels_subset = labels[is_valid_cluster]

        # 获取这些有效点在原始数组中的索引，以便最后回填颜色
        original_indices = np.where(is_valid_cluster)[0]
        # 对有效点进行排序处理
        sort_indices = np.argsort(valid_labels_subset)
        sorted_valid_labels = valid_labels_subset[sort_indices]
        sorted_valid_points = valid_points[sort_indices]
        sorted_original_indices = original_indices[sort_indices] # 同步排序索引
        # 找到每个簇的起始位置
        cluster_labels, start_indices = np.unique(sorted_valid_labels, return_index=True)
        end_indices = np.concatenate([start_indices[1:], [len(sorted_valid_labels)]])
        # print("---------------------------")

        # 4. 遍历每个有效簇
        for i, label in enumerate(cluster_labels):
            start_idx = start_indices[i]
            end_idx = end_indices[i]
            # 获取点云数据
            cluster_points_array = sorted_valid_points[start_idx:end_idx]

            # 获取该簇在原始大数组中的索引位置，用于直接赋值颜色
            cluster_original_indices = sorted_original_indices[start_idx:end_idx]
            # --- 语义识别和特征提取 ---
            category, features = self.classify_cluster_features(cluster_points_array)
            # --- 过滤逻辑 ---
            # 判断高度或体积，如果不满足条件：
            # 1. 不添加到 clustered_informations
            # 2. 不修改颜色 (保持默认的白色)

            if features['height'] < 2 or (features['volume'] < 30 and features['z_range'] < 4):
                # print(f"跳过聚类 {label} (高度:{features['height']:.2f}, 体积:{features['volume']:.2f}) - 保持白色")
                continue  # 跳过当前循环，不计算OBB，不入库，不赋色（保持白）

            # 计算OBB并入库
            obb_info = self.compute_z_aligned_obb(cluster_points_array)
            self.clustered_informations.append({
                'center': obb_info['center'],
                'extent': obb_info['extent'],
                'rotation': obb_info['rotation']
            })
            # self.logger.info("--------------------------------")
            # self.logger.info(f"类别: {category}")
            # self.logger.info(f"高度: {features['height']:.3f}, 体积: {features['volume']:.3f}, 长宽比: {features['aspect_ratio_2d']:.3f}, 线性度: {features['linearity_2d']:.3f}, 凸性: {features['convexity']:.3f}, 密度: {features['density']:.3f}")
            # self.logger.info(f"z均值: {features['z_mean']:.3f}, z标准差: {features['z_std']:.3f}, z范围: {features['z_range']:.3f},  z变异系数: {features['z_cv']:.3f}")
            # self.logger.info("--------------------------------")
            # --- 根据类别分配颜色 ---
            # 直接使用原始索引赋值给colors数组
            if category == "building":
                colors[cluster_original_indices] = [0.0, 0.0, 1.0, 1.0]  # 蓝色
            elif category == "vegetation":
                colors[cluster_original_indices] = [0.0, 1.0, 0.0, 1.0]  # 绿色
            elif category == "pole":
                colors[cluster_original_indices] = [1.0, 0.0, 0.0, 1.0]  # 红色
            else:
                colors[cluster_original_indices] = [1.0, 1.0, 0.0, 1.0]  # 黄色

        # 5. 返回完整的输入点云，以及对应的颜色数组
        # 此时：
        # - 满足条件的聚类 -> 有颜色
        # - 不满足条件被跳过的聚类 -> 白色 (初始化颜色未改变)
        # - 噪声点 (label == -1) -> 白色 (初始化颜色未改变)
        # - 点数不足的小簇 -> 白色 (初始化颜色未改变)

        return points_np, colors

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
        输入：点云
        输出：特征字典
        特征：
        1. 高度
        2. 体积
        3. 长宽比
        4. 线性度
        5. z变异系数
        6. 凸性
        7. 密度
        8. 点数
        9. z值特征（均值、标准差、范围）
        """
        if len(points) < 3:
            return None

        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(points)

        # # 计算有向包围盒（Oriented Bounding Box, OBB）
        # # OBB是能够紧密包围点云的最小旋转包围盒，相比轴对齐包围盒（AABB）更精确
        # # 它可以根据点云的形状自动调整方向，从而更准确地描述物体的尺寸和形状
        # obb = temp_pcd.get_oriented_bounding_box()
        # # extent: OBB的三个维度尺寸 [length, width, height]，按主方向排序
        # size_obb = obb.extent

        # --- 修改：使用 Z 轴对齐的 OBB ---
        obb_info = self.compute_z_aligned_obb(points)
        size_obb = obb_info['extent']
        cluster_max_z = np.max(points[:, 2])



        # ---------------------体积（Volume）- OBB包围盒的体积
        # 计算公式：体积 = length × width × height
        # 取值范围：[0, +∞) 立方米
        # 适应场景：
        #   - 建筑物：通常 > 15m³（小型建筑），大型建筑可达10000-50000m³+
        #     * 低矮建筑物：15-100m³（护栏、矮墙等）
        #     * 普通建筑：100-10000m³
        #     * 大型建筑：> 10000m³（高层建筑、大型厂房）
        #   - 植被：变化很大，10-15000m³
        #     * 单棵树：10-500m³
        #     * 树丛：500-15000m³（体积很大）
        #   - 输电塔杆：通常 < 35m³（细长结构，体积小）
        volume = np.prod(size_obb)
        # ---------------------高度（Height）- OBB的Z轴尺寸
        # 取值范围：[0, +∞) 米
        # 适应场景：
        #   - 建筑物：通常 3-50m（多层建筑），高层建筑可达100m+
        #   - 植被：0.5-30m（单棵树 2-25m，树丛可能更高）
        #   - 输电塔杆：3-60m（典型高度）
        #   - 低矮物体：0.1-2m（护栏、矮墙等）
        # height = size_obb[2] #高度
        height = cluster_max_z - self.point_min_z
        # ---------------------俯视长宽比（Aspect Ratio 2D）- XY平面上最长边与最短边的比值
        # 取值范围：[1, +∞)，1表示正方形，越大越细长
        # 适应场景：
        #   - 建筑物：通常 1.0-5.0（规则形状，如方形、矩形建筑）
        #   - 植被：1.0-8.0（单棵树接近圆形，一排树可达4-8）
        #   - 输电塔杆：通常 > 5.5（细长形，俯视图呈线状）
        #   - 围墙/护栏：可能 > 10（非常细长）
        if min(size_obb[:2]) != 0:
            aspect_ratio_2d = max(size_obb[:2]) / min(size_obb[:2])
        else:
            aspect_ratio_2d = 0

        if min(size_obb[:2]) != 0:
            # 底面对角线长度 = sqrt(长^2 + 宽^2)
            diagonal_length = np.sqrt(size_obb[0]**2 + size_obb[1]**2)
    
            # 高度比上底面对角线长度
            height_to_diagonal = size_obb[2] / diagonal_length
        else:
            height_to_diagonal = 0

        # --------------------- PCA形状特征（2D）- 主成分分析用于描述点云在XY平面上的形状特征
        # 提取XY平面坐标（忽略Z轴高度信息，专注于俯视图形状）
        points_2d = points[:, :2]
        # 计算2D点云的中心点（均值）
        mean_2d = points_2d.mean(axis=0) #2D均值
        # 计算2D协方差矩阵：描述点云在XY平面上的分布方向和离散程度
        # 协方差矩阵的特征值反映了点云沿主方向的变化量
        cov_2d = np.cov((points_2d - mean_2d).T) #2D协方差矩阵
        # 计算协方差矩阵的特征值和特征向量
        # 特征值：表示点云沿主方向的变化量（方差）
        # 特征向量：表示主方向（第一主成分是变化最大的方向）
        eig_vals_2d, _ = np.linalg.eig(cov_2d) #2D特征值
        # 对特征值进行排序（降序），提取实部（避免复数）
        # eig_vals_2d[0]：第一主成分的方差（最大变化方向）
        # eig_vals_2d[1]：第二主成分的方差（垂直于第一主成分的方向）
        eig_vals_2d = np.sort(np.real(eig_vals_2d))[::-1] #2D特征值排序
        # --------------------- 线性度（Linearity 2D）- 衡量点云在XY平面上的细长程度
        # 计算公式：线性度 = (λ1 - λ2) / λ1
        #   其中λ1是第一主成分的方差，λ2是第二主成分的方差
        # 取值范围：[0, 1]
        #   0：点云呈圆形或方形（两个主成分相当）
        #   1：点云呈完全细长形（第一主成分占绝对主导）
        # 适应场景：
        #   - 建筑物：通常 0.0-0.75（规则形状，不是细长形）
        #   - 植被：0.38-0.98（单棵树接近圆形，一排树较细长）
        #   - 输电塔杆：通常 > 0.80（细长特征明显）
        #   - 围墙/护栏：通常 > 0.90（非常细长）
        if eig_vals_2d[0] != 0:
            linearity_2d = (eig_vals_2d[0] - eig_vals_2d[1]) / eig_vals_2d[0] #线性度
        else:
            linearity_2d = 0  # 避免除零错误
       

        # 统计特征
        point_num = len(points)
        # 密度（Density）- 点云密度，单位体积内的点数
        # 计算公式：密度 = 点数 / OBB体积
        # 取值范围：[0, +∞) 点/立方米
        # 适应场景：
        #   - 建筑物：通常 0.03-20.0（实体结构，点云密集）
        #     * 低矮建筑物：可能 > 5.0（体积小但点密集）
        #     * 大型建筑物：可能 < 0.1（体积大，密度相对较低）
        #   - 植被：通常 0.03-1.5（稀疏，点云不密集）
        #     * 单棵树：0.1-1.0
        #     * 树丛：0.03-0.4（体积大，密度低）
        #   - 输电塔杆：通常 0.2-5.0（实体结构，但体积小）
        density = point_num / volume if volume != 0 else 0

        # --------------------- Z值特征（高度分布特征）- 描述点云在垂直方向上的分布特性
        z_values = points[:, 2]  # 提取所有点的z坐标
        
        # z值均值（z_mean）- 所有点的平均高度位置
        # 取值范围：取决于坐标系，通常 [-200, 200] 米（相对高度）
        # 适应场景：
        #   - 建筑物：变化很大，取决于建筑物高度和地面高度
        #   - 植被：通常接近地面高度（植被从地面开始）
        #   - 输电塔杆：通常较高（塔杆底部到顶部）
        z_mean = np.mean(z_values)  # z值均值（平均高度）
        
        # --------------------- z值标准差（z_std）- 高度分布的离散程度
        # 取值范围：[0, +∞) 米，0表示所有点在同一高度
        # 适应场景：
        #   - 建筑物：通常 0.5-8.0m（多层建筑有楼层，分布较均匀）
        #   - 植被：通常 0.4-4.0m（形状不规则，垂直分布不均匀）
        #   - 输电塔杆：通常 < height*0.3（结构规则，垂直分布均匀）
        z_std = np.std(z_values)  # z值标准差（高度分布的离散程度）
        
        # --------------------- z值范围（z_range）- 最大z值与最小z值的差，即高度跨度
        # 取值范围：[0, +∞) 米
        # 适应场景：
        #   - 建筑物：通常 > 15m（多层建筑高度跨度大）
        #     * 低矮建筑物：可能 0.5-5m
        #     * 高层建筑：可达 50-100m+
        #   - 植被：通常 1-15m（单棵树 2-25m，树丛变化大）
        #   - 输电塔杆：通常接近height（细长结构，高度跨度大）
        z_range = np.max(z_values) - np.min(z_values)  # z值范围（高度跨度）
        
        
        # --------------------- z值变异系数（z_cv）- 反映z值分布的相对离散程度（均匀性指标）
        # 计算公式：CV = z_std / |z_mean|，如果z_mean为0则使用z_range
        # 取值范围：[0, +∞)，无量纲
        # 物理意义：
        #   - 接近0：点云在z轴上分布非常均匀（如规则建筑物，各楼层点数相近）
        #   - 较大值：点云在z轴上分布不均匀（如植被，集中在某些高度）
        # 适应场景：
        #   - 建筑物：通常较小（0.05-0.3），多层建筑分布较均匀
        #   - 植被：通常较大（0.1-0.8），分布不均匀，集中在某些高度
        #   - 输电塔杆：通常较小（0.05-0.2），结构规则，分布均匀
        if abs(z_mean) > 1e-6:  # 避免z_mean接近0的情况
            z_cv = z_std / abs(z_mean)  # 变异系数
        else:
            # 如果z_mean接近0，使用z_range作为分母
            z_cv = z_std / max(z_range, 1e-6) if z_range > 1e-6 else 0

        # --------------------- 凸性特征（Convexity）- 衡量点云形状的规则程度
        # 计算公式：凸性 = 凸包体积 / OBB体积
        # 取值范围：[0, 1]
        #   1：点云完全凸形，形状规则（如立方体、球体）
        #   0：点云有严重凹陷，形状不规则
        # 适应场景：
        #   - 建筑物：通常 0.26-0.75（形状较规则）
        #     * 规则建筑：> 0.55（方形、矩形建筑）
        #     * 复杂建筑：0.26-0.55（有装饰、不规则结构）
        #   - 植被：通常 0.26-0.51（形状不规则，有凹陷）
        #     * 单棵树：0.35-0.51
        #     * 树丛：0.26-0.36（更不规则）
        #   - 输电塔杆：通常 0.3-0.6（细长但规则）
        # 物理意义：
        #   - 接近1：点云形状规则，接近凸形（如建筑物、规则物体）
        #   - 接近0：点云形状不规则，有凹陷（如植被、不规则物体）
        try:
            # 计算点云的凸包（Convex Hull）
            # 凸包是包围给定点云集合的最小凸多边形/多面体
            # compute_convex_hull()返回元组：(TriangleMesh, List[int])
            # 第一个元素是凸包的三角网格，第二个元素是点索引列表
            convex_hull_result = temp_pcd.compute_convex_hull()
            
            if convex_hull_result and len(convex_hull_result) > 0:
                hull_mesh = convex_hull_result[0]  # 获取凸包网格
                
                # 获取凸包体积
                # TriangleMesh对象有get_volume()方法计算体积
                if hasattr(hull_mesh, 'get_volume'):
                    hull_vol = hull_mesh.get_volume()
                else:
                    # 如果方法不存在，尝试使用其他方法或设为0
                    hull_vol = 0
                
                # 计算凸性：凸包体积与OBB体积的比值
                # 避免除零错误
                if volume > 0 and hull_vol > 0:
                    convexity = hull_vol / volume
                    # 凸性理论上应该在[0, 1]范围内，但可能因为数值误差略大于1
                    convexity = min(convexity, 1.0)  # 限制最大值
                else:
                    convexity = 0
            else:
                convexity = 0
        except Exception as e:
            # 捕获所有异常（计算失败、内存不足等），返回默认值0
            # 可以在这里添加日志记录异常信息，便于调试
            convexity = 0

        # --------------------- 新增：杆身线性度（Body Linearity） ---------------------
        # 过滤掉顶部和底部 20% 的点，只取中间段的杆身
        z_min = np.min(z_values)
        z_max = np.max(z_values)
        z_range = z_max - z_min
        mask_body = (z_values > (z_min + 0.2 * z_range)) & (z_values < (z_max - 0.2 * z_range))
        points_body = points[mask_body]

        if len(points_body) > 3:
            # 对杆身部分重新计算2D PCA
            points_body_2d = points_body[:, :2]
            mean_body_2d = points_body_2d.mean(axis=0)
            cov_body_2d = np.cov((points_body_2d - mean_body_2d).T)
            eig_vals_body, _ = np.linalg.eig(cov_body_2d)
            eig_vals_body = np.sort(np.real(eig_vals_body))[::-1]

            if eig_vals_body[0] != 0:
                linearity_body = (eig_vals_body[0] - eig_vals_body[1]) / eig_vals_body[0]
            else:
                linearity_body = 0
        else:
            linearity_body = 0

        # --------------------- 新增特征：Z轴分布均匀度 ---------------------
        # 物理意义：将点云在Z轴方向切成N层，看每层的点数是否一致。
        # 取值范围：[0, 1]
        #   接近 1：每层点数差不多（如电线杆、树木、完整的建筑墙面扫描） -> 均匀
        #   接近 0：点集中堆在某一层或几层（如无人机俯拍建筑顶部、车辆顶部） -> 不均匀
        # 适应场景：
        #   - 俯拍建筑：通常 < 0.3（点全在顶层）
        #   - 路侧杆状物：通常 > 0.6（垂直方向都有点）
        z_distribution_uniformity = 0.0
        # 只有当高度大于一定阈值（比如0.5米）时才计算，避免平面物体误判
        if z_range > 0:
            # 将高度方向切分为 10 层
            num_bins = 10
            # 计算每个点所属的层索引 (0 到 9)
            bin_indices = np.floor(((z_values - z_min) / z_range) * num_bins).astype(int)
            # 修正最大值可能越界的情况（比如刚好等于z_max时）
            bin_indices = np.clip(bin_indices, 0, num_bins - 1)
            # 统计每层的点数
            hist, _ = np.histogram(bin_indices, bins=num_bins, range=(0, num_bins))
            # 计算均匀度： (1 - 标准差/均值)
            # 这里我们用一个简化的归一化标准差
            mean_count = np.mean(hist)
            std_count = np.std(hist)
            if mean_count > 0:
                # 系数 0.5 是为了调整敏感度，可根据实际数据调整
                # 如果完全均匀，std=0，uniformity=1
                # 如果极度不均匀（点都在一层），std很大，uniformity接近0
                z_distribution_uniformity = 1.0 - (std_count / (mean_count + 1e-6))
                z_distribution_uniformity = max(0.0, z_distribution_uniformity)
        return {
            # 基础几何特征
            "height": height,  # 高度（米）：[0, +∞)，建筑物3-50m，植被0.5-30m，塔杆3-60m
            "volume": volume,  # 体积（立方米）：[0, +∞)，建筑物>15m³，植被变化大，塔杆<35m³
            "aspect_ratio_2d": aspect_ratio_2d,  # 长宽比：[1, +∞)，建筑物1-5，植被1-8，塔杆>5.5
            "linearity_2d": linearity_2d,  # 线性度：[0, 1]，建筑物0-0.75，植被0.38-0.98，塔杆>0.80
            "convexity": convexity,  # 凸性：[0, 1]，建筑物0.26-0.75，植被0.26-0.51，塔杆0.3-0.6
            "density": density,  # 密度（点/立方米）：[0, +∞)，建筑物0.03-20，植被0.03-1.5，塔杆0.2-5.0
            "point_num": point_num,  # 点数：[3, +∞)，取决于点云密度和物体大小
            "linearity_body": linearity_body,  # 杆身线性度
            "height_to_diagonal": height_to_diagonal,  # 高度与底面对角线长度的比值
            # Z值分布特征
            "z_mean": z_mean,  # z值均值（米）：取决于坐标系，相对高度
            "z_std": z_std,  # z值标准差（米）：[0, +∞)，建筑物0.5-8，植被0.4-4，塔杆<height*0.3
            "z_range": z_range,  # z值范围（米）：[0, +∞)，建筑物>15m，植被1-15m，塔杆接近height
            "z_cv": z_cv,  # z值变异系数：[0, +∞)，越小越均匀，建筑物0.05-0.3，植被0.1-0.8，塔杆0.05-0.2
            "z_distribution_uniformity": z_distribution_uniformity
        }


    def classify_cluster_features(self, points: np.ndarray) -> Tuple[str, dict]:
        """
        对聚类点云进行分类，返回类别和特征信息
        优化的分类逻辑，更准确地区分建筑物、植被和输电塔杆

        Returns:
            tuple: (类别名称, 特征字典)
        """
        # 提取特征
        features = self.extract_cluster_features(points)

        if features is None:
            return "unknown", {}

        h = features["height"] #高度
        v = features["volume"] #体积
        ar = features["aspect_ratio_2d"] #长宽比
        h_ar = features["height_to_diagonal"] 
        lin = features["linearity_2d"] #线性度
        con = features["convexity"] #凸性
        den = features["density"] #密度
        lin_body = features["linearity_body"] #线性度2
        z_mean = features["z_mean"] #z值均值
        z_std = features["z_std"] #z值标准差
        z_range = features["z_range"] #z值范围
        z_cv = features["z_cv"] #z值变异系数
        z_d = features["z_distribution_uniformity"]

        # print(
        #     f": H={h:.3f}, V={v:.3f}, 凸性={con:.3f}, 密度={den:.3f}, 长宽比={ar:.3f}, 线性度={lin:.3f}, 线性度2={lin_body:.3f}, 线性度3 = {h_ar:.3f}, z均值={z_mean:.3f}, z标准差={z_std:.3f}, z范围={z_range:.3f}, zcv={z_cv:.3f}, z分布 = {z_d:.3f}")

        # ========== 分类规则（按优先级排序）==========
        
        # 1. 输电塔杆（优先判断，特征最明显）
        # 核心特征：
        # - 高度：3-60米（典型塔杆高度范围）
        # - 体积：很小（< 35立方米，排除大型结构）
        # - 线性度：极高（> 0.80，细长特征明显）
        # - 长宽比：很大（> 5.5，俯视图呈细长形）
        # - 密度：中等偏高（> 0.2，实体结构）
        # - z值特征：
        #   * z_range接近height（细长结构，高度跨度大）
        #   * z_std相对较小（< height*0.3，结构规则，垂直分布均匀）
        #   * z_mean可能较高（> 5米，塔杆通常较高）
        # 排除条件：体积过大或线性度不够的物体
        # 细杆状物体判断
        # if (
        #     h < 5
        #     and v < 35.0
        #     and lin > 0.80 
        #     and ar > 1.5
        #     and den > 0.1
        #     and z_range > h * 0.7  # z范围应该接近高度（细长结构）
        #     and z_std < h * 0.3   # z标准差较小（结构规则）
        #     ) or (  # 网状输电塔杆判断
        #     h > 5
        #     # and lin > 0.80 
        #     # and ar > 1.5
        #     and den < 0.1
        #     and z_range > h * 0.7  # z范围应该接近高度（细长结构）
        #     and z_std < h * 0.3   # z标准差较小（结构规则）
        #     ):  
        #     return "pole", features

          # 细杆状物体判断
        # if (
        #     h < 8
        #     and v < 35.0
        #     and lin > 0.80 
        #     and ar > 1.5
        #     # and z_std < h * 0.3   # z标准差较小（结构规则）
        #     ):
        #     print("pole 1")
        #     return "pole", features
        
        if (
            h < 15
            and v < 50.0
            and z_range >5
            and den > 1
            and lin_body > 0.80 
            # and z_std < h * 0.3   # z标准差较小（结构规则）
            ):
            # print("pole 2")
            return "pole", features
        if(h_ar > 5 and z_range < 10):
            # print("pole 3")
            return "pole", features
        # if (h > 40 and v < 500.0): 
        #     print("pole 3")
        #     return "pole", features
        
        # 2. 建筑物（次优先判断）
        # 基于实际数据分析的特征范围：
        # Building样本特征：
        #   - height: 27.77m (很高)
        #   - volume: 47528m³ (很大)
        #   - z_range: 25.06m (很大，>15m)
        #   - z_std: 7.46m (较大)
        #   - linearity: 0.31 (较低，不是细长形)
        #   - aspect_ratio: 1.27 (适中，不是细长形)
        #   - density: 0.036 (很低，因为体积很大)
        #   - convexity: 0.26 (较低)
        # 
        # 分类策略：
        # 1. z_range > 15m 是关键特征（建筑物通常很高，z范围大）
        # 2. 排除细长形物体（可能是塔杆或一排树）
        # 3. 高度应该较高（> 10m）或体积很大（> 10000m³）
        if (
            z_range > 20.0   # z范围很大（关键特征）
            and z_std > 4
            and den > 0.01
            and lin < 0.8
            and z_cv < 0.35 #z变异系数较小（建筑物分布较均匀）
            ) :
            # print("b1")
            return "building", features

        if(
            z_d < 0.1 and z_cv < 0.35
        ):
            # print("b4")
            return "building", features
        
        if (z_range > 20 and z_cv < 0.35) : 
                # if (not (lin > 0.8 )):
                #     print(f"建筑物特征：z标准差={z_std:.3f}, z范围={z_range:.3f}")
            # print("b2")
            return "building", features
        if (h > 70): #
            # print("b3")
            return "building", features
        # if (
        #     z_range > 20.0   # z范围很大（关键特征）
        #     and z_std > 4
        #     and den > 0.01
        #     and lin < 0.8
        #     and z_cv < 0.35 #z变异系数较小（建筑物分布较均匀）
        #     )  or (h > 30):  #(z_range > 20 and z_cv < 0.35)
        #         # if (not (lin > 0.8 )):
        #         #     print(f"建筑物特征：z标准差={z_std:.3f}, z范围={z_range:.3f}")
        #         return "building", features

        # 3. 植被（最后判断，特征相对宽泛）
        # 基于实际数据分析的特征范围：
        # Vegetation样本特征范围：
        #   - height: 0.90-12.12m (范围很广)
        #   - volume: 69.72-13557.98m³ (变化很大)
        #   - z_range: 1.31-12.71m (都 < 15m)
        #   - z_std: 0.40-2.93m (相对较小)
        #   - linearity: 0.70-0.98 (较高，可能是细长形)
        #   - aspect_ratio: 1.83-4.07 (变化较大)
        #   - density: 0.03-1.04 (变化很大)
        #   - convexity: 0.26-0.51 (较低到中等)
        # 
        # 分类策略：
        # 1. z_range < 15m (排除建筑物)
        # 2. 排除已判断为塔杆的物体
        # 3. 高度在合理范围内（0.3-15m）
        if (den < 2 ):  # z范围较小（排除建筑物）
            return "vegetation", features

        # 4. 未知目标（不符合以上任何类别）
        return "unknown", features
    

