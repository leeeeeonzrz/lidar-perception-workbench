import pathlib
from pathlib import Path
from matplotlib import cm, pyplot as plt # 仅用于色彩映射表获取
import open3d as o3d
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import os
import glob
from collections import deque
import time
# 导入工具函数
import utils
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, pyqtSlot,QTimer
from PyQt5.QtCore import QMutex, QMutexLocker
import threading

from multiprocessing import Queue

class ObStacleRecog(QObject):
    signal_processed = pyqtSignal(object, object)  # 处理完成信号
    def __init__(self, parent=None):
        super(ObStacleRecog, self).__init__(parent)

        #地面分割参数
        self.ground_seg_distance_threshold = 1.0
        self.ground_seg_ransac_n = 3
        self.ground_seg_num_iterations = 10

        #聚类参数
        self.nonground_cluster_eps = 2
        self.nonground_cluster_min_points_dbscan = 15
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


    def segment_ground_ransac(self, pcd: o3d.geometry.PointCloud,
                              distance_threshold: float,
                              ransac_n: int,
                              num_iterations: int) -> tuple:
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

    def cluster_nonground_points(self, nonground_pcd: o3d.geometry.PointCloud,
                                 eps: float,
                                 min_points_dbscan: int,
                                 min_cluster_size_for_vis: int) -> o3d.geometry.PointCloud:
        """
        对非地面点云进行欧式聚类 (DBSCAN) 并为簇着色。
        返回: 带有簇颜色的点云。噪声点或过小的簇会被滤除。
        """
        if nonground_pcd is None or nonground_pcd.is_empty() or len(nonground_pcd.points) < min_points_dbscan:
            # rospy.logdebug_throttle(5.0, "cluster_nonground_points: 输入点云为空或点数不足以聚类。")
            return o3d.geometry.PointCloud()
        try:
            labels = np.array(nonground_pcd.cluster_dbscan(eps=eps, min_points=min_points_dbscan, print_progress=False))
        except RuntimeError as e:
            print( f"DBSCAN聚类错误: {e}. 返回空聚类点云。")
            return o3d.geometry.PointCloud()

        max_label = labels.max()  # 获取最大的簇ID (-1 表示噪声)
        clustered_points_list, clustered_colors_list = [], []

        if max_label >= 0:  # 如果至少存在一个有效簇 (ID >= 0)
            points_np = np.asarray(nonground_pcd.points)
            try:
                colormap = plt.get_cmap('hsv')  # 使用HSV色彩空间获取区分度较高的颜色
            except AttributeError:  # Matplotlib < 3.5
                colormap = cm.get_cmap('hsv')

            for i in range(max_label + 1):  # 遍历每个簇 (0 到 max_label)
                cluster_indices = np.where(labels == i)[0]
                if len(cluster_indices) >= min_cluster_size_for_vis:  # 只处理足够大的簇
                    cluster_points = points_np[cluster_indices]
                    # 为簇生成颜色，确保 max_label > 0 时分母不为0
                    color_idx_norm = float(i) / max(1, max_label) if max_label > 0 else 0.5
                    color = colormap(color_idx_norm)[:3]  # 获取RGB

                    clustered_points_list.append(cluster_points)
                    clustered_colors_list.append(np.tile(color, (len(cluster_points), 1)))

        if not clustered_points_list:  # 如果没有足够大的簇被保留
            # rospy.logdebug_throttle(5.0, "所有簇都太小或全是噪声点，返回空的聚类点云。")
            return o3d.geometry.PointCloud()

        final_points = np.vstack(clustered_points_list)
        final_colors = np.vstack(clustered_colors_list)

        pcd_clustered_colored = o3d.geometry.PointCloud()
        pcd_clustered_colored.points = o3d.utility.Vector3dVector(final_points)
        pcd_clustered_colored.colors = o3d.utility.Vector3dVector(final_colors)

        return pcd_clustered_colored


    def cluster_nonground_points2(self, nonground_pcd: o3d.geometry.PointCloud,
                                 eps: float,
                                 min_points_dbscan: int,
                                 min_cluster_size_for_vis: int) -> o3d.geometry.PointCloud:
        """
        对非地面点云进行欧式聚类 (DBSCAN) 并为簇着色。
        返回: 带有簇颜色的点云。噪声点或过小的簇会被滤除。
        """
        if nonground_pcd is None or nonground_pcd.is_empty() or len(nonground_pcd.points) < min_points_dbscan:
            # rospy.logdebug_throttle(5.0, "cluster_nonground_points: 输入点云为空或点数不足以聚类。")
            # return o3d.geometry.PointCloud()
            return np.empty((0, 3)), np.empty((0, 4))
        try:
            labels = np.array(nonground_pcd.cluster_dbscan(eps=eps, min_points=min_points_dbscan, print_progress=False))
        except RuntimeError as e:
            print( f"DBSCAN聚类错误: {e}. 返回空聚类点云。")
            # return o3d.geometry.PointCloud()
            return np.empty((0, 3)), np.empty((0, 4))

        max_label = labels.max()  # 获取最大的簇ID (-1 表示噪声)
        clustered_points_list, clustered_colors_list = [], []

        if max_label >= 0:  # 如果至少存在一个有效簇 (ID >= 0)
            points_np = np.asarray(nonground_pcd.points)
            try:
                colormap = plt.get_cmap('hsv')  # 使用HSV色彩空间获取区分度较高的颜色
            except AttributeError:  # Matplotlib < 3.5
                colormap = cm.get_cmap('hsv')

            for i in range(max_label + 1):  # 遍历每个簇 (0 到 max_label)
                cluster_indices = np.where(labels == i)[0]
                if len(cluster_indices) >= min_cluster_size_for_vis:  # 只处理足够大的簇
                    cluster_points = points_np[cluster_indices]

                    # 为簇生成颜色，确保 max_label > 0 时分母不为0
                    color_idx_norm = float(i) / max(1, max_label) if max_label > 0 else 0.5
                    color_rgb = colormap(color_idx_norm)[:3]  # 获取RGB
                    color = (*color_rgb, 1.0)

                    clustered_points_list.append(cluster_points)
                    clustered_colors_list.append(np.tile(color, (len(cluster_points), 1)))

        if not clustered_points_list:  # 如果没有足够大的簇被保留
            # rospy.logdebug_throttle(5.0, "所有簇都太小或全是噪声点，返回空的聚类点云。")
            # return o3d.geometry.PointCloud()
            return np.empty((0, 3)), np.empty((0, 4))  # 返回空的点数组和颜色数组

        final_points = np.vstack(clustered_points_list)
        final_colors = np.vstack(clustered_colors_list)

        pcd_clustered_colored = o3d.geometry.PointCloud()
        pcd_clustered_colored.points = o3d.utility.Vector3dVector(final_points)
        # pcd_clustered_colored.colors = o3d.utility.Vector3dVector(final_colors)

        return final_points, final_colors

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

        return center, dimensions, (min_coords, max_coords)


    @pyqtSlot(o3d.geometry.PointCloud)
    def run(self, point: o3d.geometry.PointCloud):
        start_time = time.time()
        if point.has_points():
            ground_local_map_W, nonground_local_map_W = self.segment_ground_ransac(point,self.ground_seg_distance_threshold,self.ground_seg_ransac_n,self.ground_seg_num_iterations)
            print(f"  局部地图分割 (W): 地面 {len(ground_local_map_W.points)}, 非地面 {len(nonground_local_map_W.points)}")

            ground_points_np = np.asarray(ground_local_map_W.points)
            ground_colors_np = np.array([[0.5, 0.5, 0.5, 1.0]] * len(ground_local_map_W.points))

            no_ground_points_np, no_ground_colors_np = self.cluster_nonground_points2(
                nonground_local_map_W,
                self.nonground_cluster_eps,
                self.nonground_cluster_min_points_dbscan,
                self.nonground_cluster_min_points_for_vis
            )

            merged_points = np.vstack([ground_points_np, no_ground_points_np])
            merged_colors = np.vstack([ground_colors_np, no_ground_colors_np])
            print(f"Slot received. Points: {type(merged_points)}, Colors: {type(merged_colors)}")
            # self.signal_processed.emit(merged_points, merged_colors)

            print(f"  点云总数 {len(merged_points)}")
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"代码运行时间: {elapsed_time:.4f} 秒")
            # nonground_clustered_local_map_W = self.cluster_nonground_points(
            #     nonground_local_map_W,
            #     self.nonground_cluster_eps,
            #     self.nonground_cluster_min_points_dbscan,
            #     self.nonground_cluster_min_points_for_vis
            # )
            # print(f"  局部地图非地面聚类 (W): {len(nonground_clustered_local_map_W.points)} 点 (有效簇)")

    @pyqtSlot(bool)
    def run2(self,running):
        import open3d as o3d
        import numpy as np
        print("开始运行")
        while True:
            start_time = time.time()
            # 确保获取到的是最新的一帧
            point = o3d.geometry.PointCloud()
            if not self.pcd_queue.empty():
                print("bbbbb")
                points_array = self.pcd_queue.get_nowait()

                if points_array is not None and len(points_array) > 0:
                    print(f"获取到数据，点数: {len(points_array)}")

                    # test_points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)

                    # 创建PointCloud对象
                    point.points = o3d.utility.Vector3dVector(points_array)

            if point.has_points():
                print("ccccc")
                ground_local_map_W, nonground_local_map_W = self.segment_ground_ransac(point,self.ground_seg_distance_threshold,self.ground_seg_ransac_n,self.ground_seg_num_iterations)

                print(f"  局部地图分割 (W): 地面 {len(ground_local_map_W.points)}, 非地面 {len(nonground_local_map_W.points)}")
                ground_points_np = np.asarray(ground_local_map_W.points)
                ground_colors_np = np.array([[0.5, 0.5, 0.5, 1.0]] * len(ground_local_map_W.points))

                no_ground_points_np = None
                no_ground_colors_np = None

                if len(nonground_local_map_W.points):
                    no_ground_points_np, no_ground_colors_np = self.cluster_nonground_points2(
                        nonground_local_map_W,
                        self.nonground_cluster_eps,
                        self.nonground_cluster_min_points_dbscan,
                        self.nonground_cluster_min_points_for_vis
                    )
                if no_ground_points_np is not None and len(no_ground_points_np) > 0:
                    merged_points = np.vstack([ground_points_np, no_ground_points_np])
                    merged_colors = np.vstack([ground_colors_np, no_ground_colors_np])
                else:
                    merged_points = ground_points_np
                    merged_colors = ground_colors_np
                print(f"received. Points: {type(merged_points)}, Colors: {type(merged_colors)}")
                # self.signal_processed.emit(merged_points, merged_colors)
            else:
                time.sleep(0.5)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"代码运行时间: {elapsed_time:.4f} 秒")
            print(f"aaaaa处理队列中点云的数量 ({self.pcd_queue.qsize()})")


            # nonground_clustered_local_map_W = self.cluster_nonground_points(
            #     nonground_local_map_W,
            #     self.nonground_cluster_eps,
            #     self.nonground_cluster_min_points_dbscan,
            #     self.nonground_cluster_min_points_for_vis
            # )
            # print(f"  局部地图非地面聚类 (W): {len(nonground_clustered_local_map_W.points)} 点 (有效簇)")

