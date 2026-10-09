#!/usr/bin/env python3
# -*- coding: utf-8 -*- # 指定UTF-8编码，确保中文字符正确显示

import datetime
import rospy
from sensor_msgs.msg import PointCloud2

import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QObject,QMutex, QWaitCondition

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QPushButton, QLabel, QVBoxLayout,
                             QHBoxLayout, QGroupBox, QFormLayout, QTabWidget,
                             QSplitter, QTextEdit, QMessageBox) # 导入QMessageBox
from PyQt5.QtGui import QFont, QImage, QPixmap,QPen, QColor,QPainter,qRgba,QFontMetrics
from MainWindow import Ui_MainWindow
import pyqtgraph as pg
import pyqtgraph.opengl as gl

import cv2
import serial # 保留 serial 库，用于 IMU/GPS 读取器
import csv
import queue
import sys
import os
import threading
import time
import logging # 使用标准 logging 库，与 ROS 日志并行
import open3d as o3d


# 假设这些模块与脚本在同一目录或 PYTHONPATH 中
from IMUReader import IMUReader
from GPSReader3 import GPSReader
import utils # 假设 utils 包含旋转函数

from LandRecog2 import LandingZoneDetector

from multiprocessing import Queue, Event, Barrier, Value
from multiprocessing import Process
from threading import BrokenBarrierError
import signal
import ctypes
import copy
import math

# --- 配置参数 ---
# 使用 rospy 参数，并提供清晰的默认值
SAVE_PATH = rospy.get_param('~save_path', './output/') # 数据保存基础路径 
CAMERA_INDEX = rospy.get_param('~camera_index', 0) # 相机设备索引 (cv2.VideoCapture 使用)
CAMERA_WIDTH = rospy.get_param('~camera_width', 2048) # 相机期望宽度
CAMERA_HEIGHT = rospy.get_param('~camera_height', 1536) # 相机期望高度
CAMERA_FPS = rospy.get_param('~camera_fps', 20) # 相机轮询的目标 FPS

GPS_PORT = rospy.get_param('~gps_port', '/dev/wheeltec_gps') # GPS 串口设备
GPS_BAUD_RATE = rospy.get_param('~gps_baud_rate', 9600) # GPS 波特率
IMU_PORT = rospy.get_param('~imu_port', '/dev/wheeltec_IMU') # IMU 串口设备
IMU_BAUD_RATE = rospy.get_param('~imu_baud_rate', 460800) # IMU 波特率

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 预编译 PLY 文件头模板以提高速度
_PLY_HEADER_TEMPLATE = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    "element vertex {vertex_count}\n"
    "property float x\n"
    "property float y\n"
    "property float z\n"
    "end_header\n"
)
# --- PyQt 信号定义 ---
class GuiSignals(QObject):
    # 定义各种信号，用于线程间通信，将数据从后台传递到GUI
    sig_imu_updated = pyqtSignal(dict)         # IMU 数据更新信号
    sig_gps_updated = pyqtSignal(dict)         # GPS 数据更新信号
    sig_point_cloud_updated = pyqtSignal(object) # 点云数据更新信号 (直接传递 numpy 数组)
    sig_camera_updated = pyqtSignal(object)      # 相机图像更新信号 (直接传递 numpy 数组)
    sig_log_message = pyqtSignal(str)          # 日志消息信号
    sig_recog = pyqtSignal(o3d.geometry.PointCloud)  # lq 障碍物识别信号
    sig_recog2 = pyqtSignal(bool)

# --- 相机控制器 ---
class CameraController:
    """管理相机硬件的初始化、帧捕获和资源释放。"""
    def __init__(self, index=0, width=2048, height=1536):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            # 如果相机无法打开，则抛出异常
            raise IOError(f"无法打开相机 {index}")

        # 尝试设置分辨率并记录结果
        res_w = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        res_h = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logging.info(f"尝试设置相机分辨率: {width}x{height}。设置成功: W={res_w}, H={res_h}。实际分辨率: {actual_w}x{actual_h}")

        self.frame = None # 当前帧
        self.timestamp = 0.0 # 当前帧的时间戳
        self.lock = threading.Lock() # 线程锁，用于保护帧和时间戳的访问
        self.running = True # 控制捕获循环的标志
        self.thread = threading.Thread(target=self._capture_loop, daemon=True) # 后台捕获线程

        # 跟踪最后一次成功捕获的帧和时间
        self._last_valid_frame = None
        self._last_valid_timestamp = 0.0



    def start(self):
        """启动相机捕获线程。"""
        self.running = True
        self.thread.start()
        logging.info("相机捕获线程已启动。")

    def _capture_loop(self):
        """内部线程，持续从相机硬件捕获帧。"""
        while self.running and not rospy.is_shutdown():
            ret, frame = self.cap.read()
            timestamp = time.time() # 读取后立即获取时间戳
            if ret:
                with self.lock:
                    self.frame = frame
                    self.timestamp = timestamp
                    # 更新最后有效帧的信息
                    self._last_valid_frame = frame
                    self._last_valid_timestamp = timestamp
            else:
                # 降低日志频率以避免刷屏
                rospy.logwarn_throttle(10, "无法从相机捕获帧。")

            # 尝试达到目标 FPS，相应地休眠（如果需要可以调整）
            # 简单的休眠可能无法保证精确的 FPS，因为捕获时间会有变化
            # 计算需要休眠的时间，确保至少为0
            sleep_time = max(0, (1.0 / CAMERA_FPS) - (time.time() - timestamp))
            time.sleep(sleep_time)

    def get_latest_frame(self):
        """获取最近捕获的帧及其时间戳。"""
        with self.lock:
            # 返回副本以防止外部修改共享数据
            return self.frame.copy() if self.frame is not None else None, self.timestamp

    def get_last_valid_frame(self):
        """获取最后一次成功捕获的帧及其时间戳。"""
        with self.lock:
            # 返回副本
            return self._last_valid_frame.copy() if self._last_valid_frame is not None else None, self._last_valid_timestamp

    def stop(self):
        """停止捕获线程并释放相机资源。"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2) # 等待线程结束，设置超时
        if self.cap.isOpened():
            self.cap.release() # 释放相机硬件
        logging.info("相机资源已释放。")

# --- 数据保存器 ---
class DataSaver:
    """
    处理将传感器数据（点云、传感器读数、图像）异步保存到磁盘的功能。
    使用队列来解耦数据生成和磁盘写入。
    """
    # 预编译 PLY 文件头模板以提高速度
    _PLY_HEADER_TEMPLATE = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex {vertex_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )

    def __init__(self, base_data_path, signals, queue_size=500): # 增加队列大小
        self.base_data_path = base_data_path # 数据保存的根目录
        self.session_path = None # 当前记录会话的路径（在开始记录时设置）
        self.signals = signals # PyQt信号对象，用于发送日志消息
        # 使用较大的队列以应对潜在的高数据速率
        self.queue = queue.Queue(maxsize=queue_size)
        self.running = False # 指示是否正在记录数据
        self.save_count = {'pointcloud': 0, 'sensors': 0, 'image': 0} # 统计各种类型数据的保存数量
        self.worker = threading.Thread(target=self._save_worker, daemon=True) # 后台保存工作线程
        self.worker.start()
        logging.info("DataSaver 工作线程已启动。")

    def _create_session_path(self):
        """根据当前时间创建一个新的会话目录。"""
        # 使用 年月日_时分秒 格式创建文件夹名称
        folder = os.path.join(self.base_data_path, time.strftime("%Y%m%d_%H%M%S"))
        try:
            os.makedirs(folder, exist_ok=True) # 创建会话目录
            # 在会话目录内创建子目录以组织不同类型的数据
            os.makedirs(os.path.join(folder, "pointcloud"), exist_ok=True)
            os.makedirs(os.path.join(folder, "sensors"), exist_ok=True)
            os.makedirs(os.path.join(folder, "image"), exist_ok=True)
            return folder # 返回创建的会话路径
        except OSError as e:
            rospy.logerr(f"创建会话目录 {folder} 失败: {e}")
            self.signals.sig_log_message.emit(f"错误：无法创建保存目录 {folder}")
            return None # 创建失败返回 None

    def add_data(self, data_type, timestamp, data):
        """如果正在记录，则将数据添加到保存队列。"""
        if not self.running or self.session_path is None:
            return # 如果未处于记录状态或会话路径无效，则不添加数据

        # 使用浮点数时间戳进行内部处理/命名
        # 保留原始数据格式（点云/图像为 numpy 数组，传感器为字典）
        item = {
            'type': data_type,
            'timestamp': timestamp, # 期望是浮点数时间戳 (例如 time.time() 或 ROS time)
            'data': data
        }

        try:
            # 使用 nowait 避免阻塞传感器/ROS线程
            self.queue.put_nowait(item)
        except queue.Full:
            # 如果队列已满，记录警告信息并丢弃数据
            rospy.logwarn_throttle(5, f"DataSaver 队列已满 ({self.queue.qsize()})。正在丢弃 {data_type} 数据。")

    def _save_worker(self):
        """工作线程，从队列中取出数据项并将其保存到磁盘。"""
        while not rospy.is_shutdown(): # 只要 ROS 没有关闭就持续运行
            try:
                # 阻塞等待队列中的项目，或超时（1秒）
                item = self.queue.get(timeout=1.0)

                if not self.running: # 如果在等待期间停止了记录
                     self.queue.task_done() # 标记任务完成
                     continue # 继续下一次循环

                # 将时间戳格式化为字符串（保留6位小数）用于文件名
                # timestamp_str = f"{item['timestamp']:.6f}"
                timestamp_str = item['timestamp']


                save_func = None # 选择合适的保存函数
                if item['type'] == 'pointcloud':
                    save_func = self._save_ply_binary
                elif item['type'] == 'sensors':
                    save_func = self._save_sensors
                elif item['type'] == 'image':
                    save_func = self._save_image

                if save_func:
                    try:
                        # 调用选定的保存函数
                        save_func(timestamp_str, item['data'])
                        self.save_count[item['type']] += 1 # 更新保存计数
                    except Exception as e:
                        # 记录保存失败的错误
                        rospy.logerr(f"为时间戳 {timestamp_str} 保存 {item['type']} 数据失败: {e}")
                else:
                     # 如果遇到未知的数据类型，记录警告
                     rospy.logwarn(f"保存队列中存在未知数据类型: {item['type']}")

                self.queue.task_done() # 标记此任务已完成

            except queue.Empty:
                # 队列为空，超时发生，继续下一次循环
                if not self.running and self.queue.empty():
                    # 如果已停止记录且队列为空，工作线程可以退出
                    # (尽管 daemon=True 通常会处理这个)
                    pass
                continue
            except Exception as e:
                # 捕获队列处理中的潜在错误
                rospy.logerr(f"DataSaver 工作循环中出错: {e}")
                # 尽可能确保 task_done 被调用
                try:
                    self.queue.task_done()
                except ValueError: # 如果 task_done 被多次调用，可能会发生
                    pass

    def _save_ply_binary(self, timestamp_str, points):
        """优化的二进制 PLY 保存函数。"""
        filename = f"{timestamp_str}.ply"
        # 构建点云文件的完整路径
        full_path = os.path.join(self.session_path, "pointcloud", filename)

        try:
            # 确保 points 是 NumPy 数组且为 C-连续的 float32 类型
            if not isinstance(points, np.ndarray) or points.dtype != np.float32:
                 points = np.asarray(points, dtype=np.float32)
            if not points.flags['C_CONTIGUOUS']:
                 points = np.ascontiguousarray(points)

            # 检查点云数据的维度是否正确 (N x 3)
            if points.ndim != 2 or points.shape[1] != 3:
                rospy.logerr(f"用于 PLY 保存的点云形状无效: {points.shape}")
                return

            # 动态生成文件头
            header = self._PLY_HEADER_TEMPLATE.format(vertex_count=points.shape[0])
            header_buf = header.encode('ascii') # 编码为 ASCII 字节

            # 使用带缓冲的文件写入 (64KB 缓冲区)
            with open(full_path, 'wb', buffering=65536) as f:
                f.write(header_buf) # 写入文件头
                f.write(points.tobytes()) # 使用 tobytes() 直接写入二进制数据

        except Exception as e:
            rospy.logerr(f"PLY 文件 {filename} 保存失败: {e}")
            # 可以选择重新引发异常（如果是关键错误），或者只记录并继续
            # raise

    def _save_sensors(self, timestamp_str, data):
        """将传感器字典数据保存为 CSV 文件。"""
        filename = f"{timestamp_str}_sensors.csv"
        # 构建传感器数据文件的完整路径
        full_path = os.path.join(self.session_path, "sensors", filename)

        # 展平嵌套字典 (例如: {'gps': {'lat':...}, 'imu': {'acc_x':...}} -> {'gps_lat':..., 'imu_acc_x':...})
        flat_data = {}
        for sensor, readings in data.items():
            if isinstance(readings, dict):
                for key, value in readings.items():
                    flat_data[f"{sensor}_{key}"] = value
            else: # 处理数据可能不是嵌套的情况
                flat_data[sensor] = readings

        if not flat_data: # 如果没有数据则跳过
            return

        try:
            # 使用 newline='' 防止写入多余的空行
            with open(full_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=flat_data.keys())
                writer.writeheader() # 写入 CSV 头部（字段名）
                writer.writerow(flat_data) # 写入数据行
        except Exception as e:
             rospy.logerr(f"传感器 CSV 文件 {filename} 保存失败: {e}")

    def _save_image(self, timestamp_str, image):
        """将图像 (numpy 数组) 保存为 JPG 文件。"""
        filename = f"{timestamp_str}.jpg"
        # 构建图像文件的完整路径
        full_path = os.path.join(self.session_path, "image", filename)
        try:
            # 保存前确保图像有效
            if image is None or image.size == 0:
                rospy.logwarn(f"尝试为时间戳 {timestamp_str} 保存空图像")
                return

            # 可选：指定 JPG 质量（默认通常是 95）
            # params = [cv2.IMWRITE_JPEG_QUALITY, 90]
            # success = cv2.imwrite(full_path, image, params)

            success = cv2.imwrite(full_path, image) # 保存图像
            if not success:
                 # 如果 cv2.imwrite 返回 false，记录错误
                 rospy.logerr(f"图像文件 {filename} 保存失败 (cv2.imwrite 返回 false)")
        except Exception as e:
            rospy.logerr(f"图像文件 {filename} 保存失败: {e}")

    def start(self):
        """启动数据记录会话。"""
        if self.running:
            self.signals.sig_log_message.emit("记录已在进行中。")
            return

        # 创建新的会话路径
        new_path = self._create_session_path()
        if new_path:
            self.session_path = new_path
            self.running = True
            self.save_count = {'pointcloud': 0, 'sensors': 0, 'image': 0} # 重置保存计数
            self.signals.sig_log_message.emit(f"开始记录，保存至: {self.session_path}")
            rospy.loginfo(f"数据记录已启动。路径: {self.session_path}")
        else:
            # 如果无法创建目录，通知用户
            self.signals.sig_log_message.emit("错误：无法启动记录，请检查目录权限或磁盘空间。")

    def stop(self):
        """停止记录会话并等待队列清空。"""
        if not self.running:
            # self.signals.sig_log_message.emit("记录尚未开始。")
            return

        self.running = False # 通知工作线程停止接受新项目
        self.signals.sig_log_message.emit("正在停止记录，请稍候...")
        rospy.loginfo("正在停止数据记录，等待保存队列清空...")

        # 等待队列中的所有项目被处理
        # 添加超时以防止在工作线程卡住时无限期阻塞
        qsize = self.queue.qsize()
        start_time = time.time()
        while not self.queue.empty():
            if time.time() - start_time > 30: # 30 秒超时
                 rospy.logerr(f"等待 DataSaver 队列清空超时。剩余 {self.queue.qsize()} 个项目。")
                 break
            # 降低日志频率
            rospy.loginfo_throttle(2, f"正在等待保存队列... 剩余 {self.queue.qsize()} 个项目。")
            time.sleep(0.1)

        # 另一种方法是 queue.join()，但如果工作线程过早退出，需要小心处理
        # self.queue.join() # 阻塞直到所有任务完成

        final_qsize = self.queue.qsize()
        if final_qsize == 0:
             log_msg = f"记录结束！共保存 点云: {self.save_count['pointcloud']}, 传感器: {self.save_count['sensors']}, 图像: {self.save_count['image']}"
        else:
             # 如果队列中仍有项目未保存，报告此情况
             log_msg = f"记录结束！但有 {final_qsize} 个项目未保存。共保存 点云: {self.save_count['pointcloud']}, 传感器: {self.save_count['sensors']}, 图像: {self.save_count['image']}"

        self.signals.sig_log_message.emit(log_msg) # 发送最终日志消息到GUI
        rospy.loginfo(log_msg) # 同时记录到 ROS 日志
        self.session_path = None # 停止后重置会话路径

# --- ROS 交互与传感器轮询节点 ---
# 将 ROS 处理和传感器轮询合并到一个逻辑单元中，以简化结构
# 这个类在 GUI 线程之外运行
class SensorRosNode:
    """
    管理传感器（IMU, GPS）的轮询、ROS 节点的初始化、
    点云数据的订阅和处理，以及与 DataSaver 和 GUI 的交互。
    """
    def __init__(self, signals, data_saver, camera_ctl, imu_reader, gps_reader):
        self.signals = signals # PyQt 信号，用于与 GUI 通信
        self.data_saver = data_saver # 数据保存器实例
        self.camera_ctl = camera_ctl # 相机控制器实例
        self.imu = imu_reader # IMU 读取器实例
        self.gps = gps_reader # GPS 读取器实例

        self.running = True # 控制线程运行的标志

        # 存储最新的传感器读数，供点云回调函数访问
        self.latest_imu_data = {}
        self.latest_gps_data = {}
        self.current_point_cloud = {}
        self.sensor_lock = threading.Lock() # 用于保护最新传感器数据访问的锁

        # ROS 订阅器（在此处初始化，在节点初始化后使用）
        self.pc_sub = None

        self.frame_count = 0

        self.pcd_queue = queue.Queue()
        self.pcd_queue2 = Queue() #先尝试 后面再改
        self.pcd_queue_obstacle = Queue()
        self.data_queue = Queue()


        # self.pcd_queue = Queue()
        #障碍物检测 和 着陆区 提取标志位
        self.flag_recog = False
        self.flag_landing  = False

        self.pitch_set_angle = -20
        # self.pitch_set_angle = -40 #lq

    def _poll_sensors(self):
        """轮询 IMU、GPS，并发出信号以更新 GUI。"""
        # 此函数在 start() 方法管理的单独线程中运行
        rate = rospy.Rate(20) # 轮询频率（可根据需要调整）
        while self.running and not rospy.is_shutdown():
            try:
                # lq 因为不存在imu和GPS 先注释
                # # --- IMU ---
                # 假设 update() 在有新数据时返回 True
                imu_updated = self.imu.update()
                if True:
                    current_imu_data = self.imu.get_data()
                    with self.sensor_lock: # 使用锁保护共享数据的更新
                        self.latest_imu_data = current_imu_data
                    # 仅当 GUI 可能需要更新时才发出信号（或者如果代价很小，则总是发出）
                    self.signals.sig_imu_updated.emit({'imu': current_imu_data})
                    imu_data_frame = current_imu_data.get('imu', {})
                    if imu_data_frame:
                        self.pitch_set_angle = imu_data_frame.get('pitch', 0)
                # --- GPS ---
                # 假设 update() 在有新数据时返回 True
                gps_updated = self.gps.update()
                if True:
                    # 假设 get_data() 返回一个字典或对象
                    current_gps_data = self.gps.get_data()
                    with self.sensor_lock: # 使用锁
                         self.latest_gps_data = current_gps_data
                    self.signals.sig_gps_updated.emit({'gps': current_gps_data})
                    # 可以减少日志记录频率或只记录关键信息
                    # logging.info(f"GPS 数据: 纬度={current_gps_data.get('latitude', 'N/A')}, 经度={current_gps_data.get('longitude', 'N/A')}")

                # --- 相机 ---
                # 相机帧由 CameraController 的内部线程捕获
                # 这里只获取最新的帧用于显示
                frame, _ = self.camera_ctl.get_latest_frame()
                if frame is not None:
                    # 发送信号让 GUI 更新相机画面
                    self.signals.sig_camera_updated.emit(frame)

            except serial.SerialException as e:
                 # 处理串口通信错误
                 rospy.logerr_throttle(10, f"传感器轮询期间发生串口通信错误: {e}")
                 # 可以考虑尝试重新连接或通知用户
            except Exception as e:
                # 处理轮询期间的其他意外错误
                rospy.logerr_throttle(10, f"传感器轮询期间发生意外错误: {e}")

            try:
                rate.sleep() # 按设定的频率休眠
            except rospy.ROSTimeMovedBackwardsException:
                # 处理 ROS 时间回跳的情况
                rospy.logwarn("ROS 时间向后移动，忽略休眠。")
            except rospy.ROSInterruptException:
                # 处理 ROS 中断（例如 Ctrl+C）
                logging.info("在传感器轮询休眠期间收到 ROS 中断。")
                break # 如果 ROS 关闭，则退出循环


    def _ros_type_to_numpy(self, ros_type):
        """辅助函数，将 ROS PointField 类型映射到 numpy dtypes。"""
        type_map = {
            1: np.int8,    # INT8
            2: np.uint8,   # UINT8
            3: np.int16,   # INT16
            4: np.uint16,  # UINT16
            5: np.int32,   # INT32
            6: np.uint32,  # UINT32
            7: np.float32, # FLOAT32
            8: np.float64, # FLOAT64
        }
        return type_map.get(ros_type, np.uint8) # 如果未知，默认为 uint8
    
    def get_latest_frame(self):
        """获取最近捕获的帧及其时间戳。"""
        with self.sensor_lock:
            # 返回副本以防止外部修改共享数据
            return self.current_point_cloud.copy() if self.current_point_cloud is not None else None
        
    def _pointcloud_callback(self, msg: PointCloud2):
        """处理传入的 PointCloud2 消息的回调函数。"""
        try:
            # 使用 ROS 消息头中的时间戳以获得更好的精度
            # timestamp = msg.header.stamp.to_sec()

            # 获取当前时间
            now = datetime.datetime.now()
            # 将当前时间转换为年月日时分秒毫秒格式
            timestamp = now.strftime("%Y%m%d%H%M%S") + f"{now.microsecond // 1000:03d}"  # 毫秒部分补零到3位
            

            # --- 高效的 PointCloud2 到 NumPy 转换 ---
            # 如果可能，缓存字段信息（尽管 PointCloud2 格式很少在流中更改）
            field_names = [field.name for field in msg.fields]
            if 'x' not in field_names or 'y' not in field_names or 'z' not in field_names:
                rospy.logwarn_throttle(10, "PointCloud2 消息缺少 x, y, 或 z 字段。")
                return

            # 动态创建结构化 dtype 描述
            dtype_list = []
            expected_offset = 0
            for field in msg.fields:
                 # 如果字段偏移量与预期不符，说明存在填充
                 if field.offset != expected_offset:
                    padding_size = field.offset - expected_offset
                    # 添加填充描述 ('', 'V[size]')
                    dtype_list.append(('', f'V{padding_size}'))
                    expected_offset += padding_size

                 # 添加字段本身的数据类型描述
                 field_dtype = self._ros_type_to_numpy(field.datatype)
                 # 处理 count > 1 的情况 (虽然在 XYZ 中不常见)
                 itemsize = np.dtype(field_dtype).itemsize * (field.count if field.count > 0 else 1)
                 dtype_list.append((field.name, field_dtype, (field.count if field.count > 1 else 1))) # 使用正确的格式
                 expected_offset += itemsize


            # 如果 point_step 大于计算的总偏移量，添加末尾填充
            if msg.point_step > expected_offset:
                 padding_size = msg.point_step - expected_offset
                 dtype_list.append(('', f'V{padding_size}'))

            # 创建结构化的 NumPy dtype
            # 注意：上面构造 dtype_list 的方式可能需要调整，直接使用 PointCloud2 提供的字段信息更可靠
            # 使用 sensor_msgs.point_cloud2 可能更简单，但我们坚持手动转换以提高效率
            # 重新审视：直接使用 numpy.frombuffer 配合精心构造的 dtype 是高效的
            # 构造 dtype 的简化方式（假设 XYZ 是 float32 且紧密排列，需要根据实际消息调整）
            dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)]) # 简化假设
            # 检查 PointCloud2 字段以构建更精确的 dtype
            if msg.fields[0].name == 'x' and msg.fields[1].name == 'y' and msg.fields[2].name == 'z' and \
               msg.fields[0].datatype == 7 and msg.fields[1].datatype == 7 and msg.fields[2].datatype == 7 and \
               msg.point_step >= 12: # 至少包含 3 个 float32
                # 构造一个仅包含 x, y, z 的 dtype，依赖 offsets
                dtype_desc = {'names': ['x', 'y', 'z'],
                              'formats': [np.float32, np.float32, np.float32],
                              'offsets': [msg.fields[0].offset, msg.fields[1].offset, msg.fields[2].offset],
                              'itemsize': msg.point_step} # itemsize 很重要
                try:
                    dtype = np.dtype(dtype_desc)
                except ValueError as e:
                     rospy.logerr_throttle(10, f"构建 numpy dtype 失败: {e}. Point step: {msg.point_step}, Fields: {[(f.name, f.offset, f.datatype) for f in msg.fields]}")
                     return # 如果无法构建 dtype，则无法处理

            else:
                rospy.logwarn_throttle(10, "PointCloud2 字段布局不符合预期的 XYZ float32 格式，跳过处理。")
                return # 格式不符则跳过

            # 确保数据长度是 point_step 的整数倍
            num_points = len(msg.data) // msg.point_step
            if num_points == 0:
                return # 没有有效的点

            valid_data_length = num_points * msg.point_step
            data_buffer = msg.data[:valid_data_length]

            # 从缓冲区创建结构化数组
            structured_array = np.frombuffer(data_buffer, dtype=dtype)

            # 提取 x, y, z 到一个简单的 N x 3 数组（确保是 float32）
            points = np.vstack([
                structured_array['x'],
                structured_array['y'],
                structured_array['z']
            ]).T.astype(np.float32) # 确保最终是 N x 3 的 float32 数组
            # --- 转换结束 ---

            # 可选：检查最小点数
            if points.shape[0] < 100:
                # rospy.loginfo_throttle(5, f"接收到的点云点数过少: {points.shape[0]}")
                return

            # --- 发出信号以更新 GUI（下采样以提高性能）---
            # 下采样率
            downsample_factor = 5
            # points2 = utils.rotate_points_around_x_axis(points, -20) #lq
            points2 = utils.rotate_points_around_x_axis(points, self.pitch_set_angle)  # lq
            # points2 = points
            if not self.flag_recog and not self.flag_landing:  #不做任何算法上的处理，只显示点云
                self.signals.sig_point_cloud_updated.emit(points2[::downsample_factor])#lq
            self.current_point_cloud = points
            self.frame_count += 1

            if self.flag_recog or self.flag_landing:
                if self.data_queue.qsize() > 10:  # 队列累积10清空
                    while not self.data_queue.empty():
                        self.data_queue.get_nowait()
                self.data_queue.put_nowait(points2)

            #
            #
            # if self.flag_recog and not self.flag_landing: # 只测障碍物 不测盲降
            #     if self.pcd_queue_obstacle.qsize() > 10:  # 队列累积10清空
            #         while not self.pcd_queue_obstacle.empty():
            #             self.pcd_queue_obstacle.get_nowait()
            #     self.pcd_queue_obstacle.put_nowait(points2[::downsample_factor])
            #     print(f"收数据线程队列数量 ({self.pcd_queue_obstacle.qsize()})。")
            #
            # if self.flag_landing and not self.flag_recog: # 只测盲降 不测障碍物
            #     if self.pcd_queue2.qsize() > 10:  # 队列累积10清空
            #         while not self.pcd_queue2.empty():
            #             self.pcd_queue2.get_nowait()
            #     self.pcd_queue2.put_nowait(points2)
            #
            #     # print(f"盲降收数据线程队列数量 ({self.pcd_queue2.qsize()})。")
            #
            # if self.flag_landing and self.flag_recog: # 测盲降 又测障碍物
            #     if self.pcd_queue2.qsize() > 10:  # 队列累积10清空
            #         while not self.pcd_queue2.empty():
            #             self.pcd_queue2.get_nowait()
            #     self.pcd_queue2.put_nowait(points2)
            #
            #     if self.pcd_queue_obstacle.qsize() > 10:  # 队列累积10清空
            #         while not self.pcd_queue_obstacle.empty():
            #             self.pcd_queue_obstacle.get_nowait()
            #     self.pcd_queue_obstacle.put_nowait(points2[::downsample_factor])

                # print(f"盲降收数据线程队列数量 ({self.pcd_queue2.qsize()})。")

            # --- 如果正在录制，则保存数据 ---
            if self.data_saver.running:
                # 保存点云
                self.data_saver.add_data(data_type='pointcloud', timestamp=timestamp, data=points)

                # 获取与此点云时间关联的最新传感器数据
                # 这*不是*完美的同步，但使用了最近的传感器读数
                with self.sensor_lock: # 使用锁访问共享数据
                    sensors_to_save = {
                        'imu': self.latest_imu_data.copy(), # 保存副本
                        'gps': self.latest_gps_data.copy() # 保存副本
                    }
                self.data_saver.add_data(data_type='sensors', timestamp=timestamp, data=sensors_to_save)

                # 获取最后一次有效的相机帧（时间上在点云之前的最近帧）
                # 更高级的同步会缓冲帧并找到绝对最接近的帧
                frame, frame_ts = self.camera_ctl.get_last_valid_frame()
                if frame is not None:
                     # 这里可以添加检查：abs(frame_ts - timestamp) < threshold
                     # 使用点云的时间戳保存图像，以保持关联性
                     self.data_saver.add_data(data_type='image', timestamp=timestamp, data=frame)

        except Exception as e:
            # 记录处理点云回调时发生的任何错误，并包含堆栈跟踪
            rospy.logerr(f"处理点云回调时出错: {e}", exc_info=True)

    def start(self):
        """初始化 ROS 节点并启动传感器轮询和 ROS 订阅器。"""
        try:
            # 首先启动传感器轮询线程
            self.sensor_poll_thread = threading.Thread(target=self._poll_sensors, daemon=True)
            self.sensor_poll_thread.start()
            logging.info("传感器轮询线程已启动。")

            # 初始化 ROS 节点
            # 将节点初始化移到这里，以保持 ROS 逻辑集中
            # 检查 ROS 是否已初始化，避免重复初始化
            if not rospy.is_shutdown() and not rospy.core.is_initialized():
                 # 使用 anonymous=True 避免节点名称冲突
                 rospy.init_node('multisensor_capture_node', anonymous=True, log_level=rospy.INFO)
                 logging.info("ROS 节点已初始化。")

            # 节点初始化后设置 ROS 订阅器
            self.pc_sub = rospy.Subscriber(
                "/lslidar_point_cloud", # 如果需要，将话题名称设为参数
                PointCloud2,
                self._pointcloud_callback, # 指定回调函数
                queue_size=5,  # 减小队列大小，如果处理落后则更快处理或丢弃
                buff_size=2**24 # 16MB 缓冲区，根据点云大小/速率调整
            )
            logging.info(f"已订阅话题: {self.pc_sub.resolved_name}")

            rospy.spin() # 保持此线程活动以处理 ROS 回调

        except rospy.ROSInterruptException:
            logging.info("收到 ROS 关闭请求。")
        except Exception as e:
             # 记录启动过程中的严重错误
             rospy.logfatal(f"启动 SensorRosNode 时发生严重错误: {e}", exc_info=True)
        finally:
            # 确保在退出时调用 stop
            self.stop()

    def stop(self):
        """停止节点和轮询线程。"""
        self.running = False # 设置标志以停止循环
        # 等待传感器轮询线程结束
        if hasattr(self, 'sensor_poll_thread') and self.sensor_poll_thread.is_alive():
            self.sensor_poll_thread.join(timeout=1)
        # ROS 关闭通常由外部处理（Ctrl+C 或主窗口关闭）
        logging.info("SensorRosNode 已停止。")
    # --- 辅助函数：将文本渲染为 RGBA 图像 ---
def render_text_to_image(text, size=50, text_color=(255, 255, 255, 200), bg_color=(0, 0, 0, 0)):
    """将文本渲染为透明背景的 RGBA NumPy 数组"""
    font = QFont()
    font.setPointSize(size)
    # 使用 QFontMetrics 估算所需图像大小
    fm = QFontMetrics(font)
    text_width = fm.horizontalAdvance(text)
    text_height = fm.height()

    # 创建一个足够大的 QImage
    img = QImage(text_width + 2, text_height + 2, QImage.Format_ARGB32) # 加一点边距
    img.fill(qRgba(*bg_color)) # 填充背景色 (透明)

    # 在 QImage 上绘制文本
    painter = QPainter(img)
    painter.setFont(font)
    pen = QPen(QColor(*text_color)) # 设置文本颜色和透明度
    painter.setPen(pen)
    # painter.drawText(img.rect(), QtCore.Qt.AlignCenter, text) # 居中绘制
    painter.drawText(1, fm.ascent() + 1, text) # 左上角绘制 (考虑字体基线)
    painter.end()

    # 将 QImage 转换为 NumPy 数组 (BGRA -> RGBA)
    ptr = img.constBits()
    ptr.setsize(img.byteCount())
    arr = np.array(ptr).reshape(img.height(), img.width(), 4) # BGRA format

    # 转换为 RGBA
    arr = arr[:, :, [2, 1, 0, 3]] # Swap B and R

    # 返回处理好的图像数据 (可能需要转置，取决于GLImageItem期望的格式)
    # GLImageItem 通常期望 (width, height, 4)，但 numpy 数组是 (height, width, 4)
    # 我们将在创建 GLImageItem 时处理它，或者在这里转置
    return np.transpose(arr, (1, 0, 2)) # 转置为 (width, height, 4)

class GLViewZAxisController:
    """通过钩住鼠标事件将旋转限制为绕 Z 轴。"""

    def __init__(self, view: gl.GLViewWidget):
        self._view = view
        self._orig_mouse_press = view.mousePressEvent
        self._orig_mouse_move = view.mouseMoveEvent
        self._orig_mouse_release = view.mouseReleaseEvent
        self._locked_elevation = view.opts.get('elevation', 0)
        self._dragging = False
        self._last_pos = None
        self._current_azimuth = view.opts.get('azimuth', 0)

        view._z_axis_controller = self
        view.mousePressEvent = self._mouse_press_event
        view.mouseMoveEvent = self._mouse_move_event
        view.mouseReleaseEvent = self._mouse_release_event

    def _mouse_press_event(self, ev):
        if ev.button() == Qt.LeftButton and ev.modifiers() == Qt.NoModifier:
            self._locked_elevation = self._view.opts.get('elevation', self._locked_elevation)
            self._current_azimuth = self._view.opts.get('azimuth', self._current_azimuth)
            self._dragging = True
            self._last_pos = ev.pos()
            self._view.mousePos = ev.pos()
        self._orig_mouse_press(ev)

    def _mouse_move_event(self, ev):
        mouse_enabled = getattr(self._view, 'mouseEnabled', True)
        if (
            self._dragging
            and mouse_enabled
            and (ev.buttons() & Qt.LeftButton)
            and ev.modifiers() == Qt.NoModifier
        ):
            if self._last_pos is not None:
                diff = ev.pos() - self._last_pos
                self._current_azimuth -= diff.x() * 0.5
                self._view.opts['azimuth'] = self._current_azimuth
                self._view.opts['elevation'] = self._locked_elevation
                self._view.update()
            self._last_pos = ev.pos()
            ev.accept()
            return

        self._orig_mouse_move(ev)

    def _mouse_release_event(self, ev):
        if ev.button() == Qt.LeftButton:
            self._dragging = False
            self._last_pos = None
        self._orig_mouse_release(ev)


# --- 主 GUI 窗口 ---
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self,app, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.app = app
        self.signals = GuiSignals() # 初始化信号对象

        # --- 初始化硬件/读取器 ---
        # 将初始化包装在 try-except 块中以增强鲁棒性
        try:
            rospy.init_node('multisensor_capture_node', anonymous=True, log_level=rospy.INFO)
            logging.info("ROS 节点已初始化。")
        except Exception as e:
            logging.error(f"ROS 节点初始化失败: {e}", exc_info=True)

        try:
            self.camera_ctl = CameraController(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)
            self.camera_ctl.start() # 启动相机的内部捕获线程
        except Exception as e:
            logging.error(f"初始化相机失败: {e}", exc_info=True)
            self.camera_ctl = None # 如果失败，则设置为 None
            # 显示错误消息给用户
            QMessageBox.critical(self, "硬件错误", f"无法初始化相机: {e}")

        try:
            self.imu_reader = IMUReader(IMU_PORT, IMU_BAUD_RATE)
        except Exception as e:
            logging.error(f"初始化 IMU 失败: {e}", exc_info=True)
            self.imu_reader = None
            QMessageBox.critical(self, "硬件错误", f"无法初始化 IMU ({IMU_PORT}): {e}")

        try:
            self.gps_reader = GPSReader(GPS_PORT, GPS_BAUD_RATE)
            # 假设 GPSReader 内部处理连接（如果需要）
            # self.gps_reader.connect()
        except Exception as e:
            logging.error(f"初始化 GPS 失败: {e}", exc_info=True)
            self.gps_reader = None
            QMessageBox.critical(self, "硬件错误", f"无法初始化 GPS ({GPS_PORT}): {e}")

        # --- 初始化数据保存器 ---
        self.data_saver = DataSaver(SAVE_PATH, self.signals)


        # --- 初始化 ROS/传感器节点 ---
        # 将初始化好的读取器传递给节点
        self.ros_sensor_node = SensorRosNode(
            self.signals,
            self.data_saver,
            self.camera_ctl,
            self.imu_reader,
            self.gps_reader
        )


        #着陆区检测进程
        self.stop_evt = Event()
        self.flag_land_evt = Event()
        self.proc = None
        QTimer.singleShot(500, self.start_process)
        self.input_land_queue = Queue() #着陆区检测输入队列
        self.land_queue = Queue() #着陆区检测结果队列
        # 设置定时器检查结果队列
        self.timer_land = QTimer()
        self.timer_land.timeout.connect(self.check_land_results)
        self.timer_land.start(50)  # 每50毫秒检查一次
        # 存储圆圈对象的引用
        self.landing_zone_circle = None
        self.landing_zone_circle_delete = False

        # 障碍物检测进程
        self.obstacle_stop_evt = Event()
        self.flag_recog_evt = Event()
        #语义标志位
        self.flag_semantics_evt = Event()
        self.obstacle_proc = None
        self.input_obstacle_queue = Queue()
        self.obstacle_res_queue = Queue()

        #数据分发器进程
        self.data_stop_evt = Event()
        self.data_proc = None

        # # 同步屏障：2个进程需要同步，设置5秒超时防止死锁
        # self.process_barrier = Barrier(2, timeout=1.0)

        # self.shared_frame_id = Value('i', 0) #共享的帧ID计数器
        self.shared_frame_id = Value(ctypes.c_uint64, 0)  # 共享的帧ID计数器
        self.result_cache = {} #主线程中的结果缓存
        self.last_processed_frame_id = -1



        # --- 在单独的线程中启动 ROS 节点 ---
        # 启动前检查读取器是否有效
        if self.imu_reader and self.gps_reader and self.camera_ctl:#lq 因为不存在imu和GPS 先注释
        # if self.camera_ctl:
            self.node_thread = threading.Thread(target=self.ros_sensor_node.start, daemon=True)
            self.node_thread.start()
        else:
             # 如果有传感器未初始化，通知用户并可能禁用某些功能
             self.signals.sig_log_message.emit("错误：部分传感器未初始化，ROS 节点未启动。")
             # 可以禁用录制按钮等

        # --- 初始化用户界面 ---
        self.init_ui()


        # --- 连接信号与槽 ---
        self.connect_signals()

        self.recording = False # 用于跟踪录制状态
        self.process_sensor = False #用于标志是否处理点云

        self.checkBox_process_cloud.hide()
        #将语义识别先设为不可用
        self.checkBox_semantics.setEnabled(False)

    def init_ui(self):
        # """初始化主窗口的用户界面布局和控件。"""
        # self.setWindowTitle("多传感器同步采集系统 V2.0")
        # self.setGeometry(100, 100, 1600, 900) # 设置窗口位置和大小

        # --- 样式表设置 ---
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px; /* 为标题留出空间 */
                padding-top: 15px; /* 增加内边距，避免内容与标题重叠 */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left; /* 标题位置在左上角 */
                left: 10px; /* 标题距离左边框的距离 */
                padding: 0 3px; /* 标题的内边距 */
            }
            QLabel {
                min-height: 22px;
                padding: 3px;
                font-size: 10pt; /* 稍大字体 */
            }
            QTextEdit {
                background-color: #ffffff; /* 白色背景 */
                border: 1px solid #cccccc;
                border-radius: 3px;
                font-family: Consolas, Courier New, monospace; /* 等宽字体，适合日志 */
            }
            QPushButton {
                padding: 8px 15px; /* 按钮内边距 */
                font-size: 10pt;
                border: 1px solid #adadad;
                border-radius: 4px;
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                                  stop: 0 #f6f7fa, stop: 1 #dadbde); /* 默认渐变背景 */
            }
            QPushButton:hover { /* 鼠标悬停效果 */
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                                  stop: 0 #e6e7ea, stop: 1 #c8c9ce);
            }
            QPushButton:pressed { /* 鼠标按下效果 */
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                                  stop: 0 #c8c9ce, stop: 1 #e6e7ea);
            }
            QPushButton#btn_record { /* 录制按钮的特定样式 */
                font-weight: bold;
                color: white; /* 白色文字 */
                /* 初始样式 (红色代表未录制) */
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                                  stop: 0 #e57373, stop: 1 #d32f2f);
            }
            QPushButton#btn_record:hover {
                 background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                                  stop: 0 #ef9a9a, stop: 1 #e53935);
            }
             QPushButton#btn_record[recording="true"] { /* 录制状态下的样式 */
                /* 绿色代表正在录制 */
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                                  stop: 0 #81c784, stop: 1 #4caf50);
            }
            QPushButton#recordButton[recording="true"]:hover {
                 background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                                  stop: 0 #a5d6a7, stop: 1 #66bb6a);
            }
            QSplitter::handle { background-color: #d0d0d0; } /* 分割条样式 */
            QSplitter::handle:horizontal { width: 5px; }
            QSplitter::handle:vertical { height: 5px; }
        """)

        self.btn_record.toggled.connect(self.toggle_recording) # 连接状态切换信号
        self.btn_snap.clicked.connect(self.save_single_frame)

        # -- 点云视图 --
        self.gl_view = gl.GLViewWidget() # 3D 视图控件
        #GLViewZAxisController(self.gl_view) #xlh
        self.point_cloud = gl.GLScatterPlotItem(size=2, pxMode=True) # 散点图项，初始点大小为2像素
        self.gl_view.addItem(self.point_cloud) # 添加点云项到视图

        # 添加网格和坐标轴以提供参考
        grid = gl.GLGridItem()
        grid.setSize(50, 50, 1) # 设置网格大小
        grid.setSpacing(2, 2, 1) # 设置网格间距
        self.gl_view.addItem(grid)
        axis = gl.GLAxisItem()
        axis.setSize(5, 5, 5) # 设置坐标轴大小（相对于网格较小）
        self.gl_view.addItem(axis)
        
        # --- 创建距离参考圆圈 ---
        max_radius=600
        radius_step=50
        # self.add_distance_circles(max_radius, radius_step)
        self.add_distance_circles_and_labels(max_radius, radius_step)

        # 设置初始相机视角
        self.gl_view.setCameraPosition(distance=30, elevation=20, azimuth=-45)
        self.pc_group_layout.addWidget(self.gl_view) # 将 3D 视图添加到布局

        self.coord_label = QtWidgets.QLabel(self.gl_view)
        self.coord_label.setStyleSheet("""
              QLabel {
                  color: white;
                  background-color: rgba(0, 0, 0, 150);
                  padding: 5px;
                  border-radius: 3px;
                  font: bold 14px;
              }
          """)
        self.coord_label.setText(" ")
        self.coord_label.move(10, 10)  # 左上角位置
        self.coord_label.adjustSize()
        self.coord_label.show()

        # self.btn_process_cloud.clicked.connect(self.process_sensor_data)
        self.checkBox_process_cloud.clicked.connect(self.on_checkBox_process_cloud_clicked) #lq 将原来的分割地面的pushbutton改为radiobutton
        self.checkBox_obstacle_recog.clicked.connect(self.on_checkBox_obstacle_recog_clicked)
        self.checkBox_landing.clicked.connect(self.on_checkBox_landing_recog_clicked)
        self.checkBox_semantics.clicked.connect(self.on_checkBox_semantics_clicked)

        # --- FPS 计时器相关变量 ---
        self.pointcloud_count = 0 # 点云帧数计数器
        self.last_pc_time = time.time() # 上次点云 FPS 计算时间-
        self.image_count = 0 # 图像帧数计数器
        self.last_image_time = time.time() # 上次图像 FPS 计算时间


    def connect_signals(self):
        """连接所有信号到对应的槽函数。"""
        self.signals.sig_imu_updated.connect(self.update_imu)
        self.signals.sig_gps_updated.connect(self.update_gps)
        self.signals.sig_point_cloud_updated.connect(self.update_pointcloud_display)
        self.signals.sig_camera_updated.connect(self.update_camera_display)
        self.signals.sig_log_message.connect(self.append_log)

    # 使用 @QtCore.pyqtSlot 装饰器明确指定这是一个槽函数
    @QtCore.pyqtSlot(str)
    def append_log(self, message):
        """将一条消息追加到日志 QTextEdit 控件中。"""
        # 获取当前时间，格式化到毫秒
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.log_text.appendPlainText(f"[{timestamp}] {message}") # 追加消息
        # 自动滚动到底部
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    @QtCore.pyqtSlot(object)
    def update_camera_display(self, frame):
        """更新相机视图 QLabel 控件中的图像。"""
        if frame is None or not hasattr(self, 'camera_label') or self.camera_label is None:
             # 如果帧无效或标签未初始化，则返回
             return
        try:

            # 在图像上绘制红色线框
            h, w = frame.shape[:2]
            box_width = w  # 线框宽度为图像宽度的1/4
            box_height = h // 4  # 线框高度与图像高度一致

            # 计算线框的左上角和右下角坐标
            x1 = (w - box_width) // 2
            y1 = (h - box_height) // 2
            x2 = x1 + box_width
            y2 = y1 + box_height

            # 绘制红色矩形框，线宽为2像素
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 高效地将 OpenCV BGR 图像转换为 QImage (RGB)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 将 QImage 转换为 QPixmap 并进行平滑缩放以适应标签大小，保持纵横比
            pixmap = QPixmap.fromImage(q_img)
            # 使用标签的当前尺寸作为缩放目标
            scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.camera_label.setPixmap(scaled_pixmap) # 在标签上显示图像

            # 更新图像 FPS 计数器
            self.image_count += 1
            current_time = time.time()
            elapsed = current_time - self.last_image_time
            if elapsed >= 1.0: # 每秒更新一次 FPS 显示
                fps = math.ceil(self.image_count / elapsed)
                self.image_fps_label.setText(f"图像 FPS: {fps}")
                self.image_count = 0 # 重置计数器
                self.last_image_time = current_time # 更新时间戳
        except Exception as e:
            # 记录更新相机显示时发生的错误（降低日志级别）
            rospy.logwarn(f"更新相机显示时出错: {e}", exc_info=False)
            # 可以选择在标签本身显示错误消息
            # self.camera_label.setText("显示帧时出错")


    @QtCore.pyqtSlot(dict)
    def update_imu(self, data):
        """更新界面上显示 IMU 数据的标签。"""
        imu_data = data.get('imu', {}) # 安全地获取 IMU 数据字典
        if not imu_data: return # 如果没有数据则返回


        self.imu_labels_pose.setText(
            f"{imu_data.get('roll', 0):.2f}, {imu_data.get('pitch', 0):.2f}, {imu_data.get('yaw', 0):.2f}")
        self.imu_labels_accel.setText(
            f"X: {imu_data.get('acc_x', 0):.2f}, Y: {imu_data.get('acc_y', 0):.2f}, Z: {imu_data.get('acc_z', 0):.2f}")
        self.imu_labels_gyro.setText(
            f"X: {imu_data.get('gyro_x', 0):.2f}, Y: {imu_data.get('gyro_y', 0):.2f}, Z: {imu_data.get('gyro_z', 0):.2f}")
        # 检查 IMUReader 返回的磁场键名是否匹配
        self.imu_labels_mag.setText(
             f"X: {imu_data.get('norm_mag_x', 0):.2f}, Y: {imu_data.get('norm_mag_y', 0):.2f}, Z: {imu_data.get('norm_mag_z', 0):.2f}")
        self.imu_labels_quat.setText(
            f"{imu_data.get('q0', 0):.3f}, {imu_data.get('q1', 0):.3f}, {imu_data.get('q2', 0):.3f}, {imu_data.get('q3', 0):.3f}")


    @QtCore.pyqtSlot(dict)
    def update_gps(self, data):
        # """更新界面上显示 GPS 数据的标签。"""
        gps_data = data.get('gps', {}) # 安全地获取 GPS 数据字典
        if not gps_data: return


        self.gps_labels_sat.setText(f"{gps_data.get('satellites', '-')}")
        # 小心格式化经纬度，处理可能缺失的单位
        lat = gps_data.get('latitude', '-')
        lon = gps_data.get('longitude', '-')
        lat_unit = gps_data.get('unitlat', '') # 获取纬度单位
        lon_unit = gps_data.get('unitlon', '') # 获取经度单位
        self.gps_labels_lat.setText(f"{lat}{lat_unit}")
        self.gps_labels_lon.setText(f"{lon}{lon_unit}")
        self.gps_labels_alt.setText(f"{gps_data.get('altitude', '-')}")


    def add_distance_circles_and_labels(self, max_radius, step, num_points=100, label_height=5.0):
        """在 XY 平面添加同心圆和距离标签。"""
        radii = np.arange(step, max_radius + step, step)
        circle_color = (0.5, 0.5, 0.5, 0.6) # 半透明灰色
        z_offset = 0.0 # 圆圈的 Z 高度
        label_z_offset = 1.0 # 标签稍微抬高一点，避免和圆圈重叠
        label_text_color = (200, 200, 200, 200) # 标签文本颜色 (浅灰，半透明)
        label_font_size = 15 # 渲染文本时的字体大小 (像素)

        for radius in radii:
            # --- 绘制圆圈 ---
            theta = np.linspace(0, 2 * np.pi, num_points)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = np.full(num_points, z_offset)
            circle_points = np.vstack([x, y, z]).transpose()
            circle_line = gl.GLLinePlotItem(pos=circle_points, color=circle_color, width=1, mode='line_strip')
            self.gl_view.addItem(circle_line)

            # --- 创建距离标签 ---
            label_text = f"{int(radius)}m"
            try:
                # 1. 渲染文本到图像
                # 注意：render_text_to_image 返回 (width, height, 4)
                img_data = render_text_to_image(label_text, size=label_font_size, text_color=label_text_color)
                if img_data is None or img_data.size == 0:
                    print(f"警告：无法为半径 {radius} 渲染文本图像。")
                    continue

                img_width_px, img_height_px = img_data.shape[0], img_data.shape[1]

                # 2. 创建 GLImageItem
                # GLImageItem 需要 (height, width, 4) 或 (height, width)，所以需要再次转置
                label_img_item = gl.GLImageItem(np.transpose(img_data, (1, 0, 2)), glOptions='translucent')

                # 3. 计算标签在世界坐标中的尺寸
                # 我们希望标签大致有固定的世界高度，例如 label_height
                # 宽度根据图像的宽高比自动确定
                world_height = label_height
                world_width = world_height * (img_width_px / img_height_px)

                # 4. 定位和定向标签
                # 清除默认变换
                label_img_item.resetTransform()

                # a. 缩放: 将图像像素尺寸映射到世界坐标尺寸
                # GLImageItem 的 scale 是基于其内部单位（通常是像素）
                # 我们需要设置它的 'width' 和 'height' 属性（如果可用）或通过变换缩放
                # 一个简单的方法是先应用缩放变换
                # 注意：GLImageItem 可能没有直接的 width/height 属性，需要用 transform
                label_img_item.scale(world_width / img_width_px, world_height / img_height_px, 1.0, local=False)


                # b. 旋转: 使其平行于 Y 轴 (绕 Z 轴旋转 90 度)
                label_img_item.rotate(-90, 0, 0, 1, local=True) # 绕 Z 轴旋转

                # c. 平移: 移动到圆圈旁边 (例如，放在正 X 轴上，中心对齐半径点)
                # 平移量需要考虑旋转后的坐标系和缩放
                # 我们希望标签的中心（或底部）位于 (radius, 0, label_z_offset)
                # 由于我们先缩放再旋转，平移量应为目标世界坐标
                # 标签的局部原点是左下角，旋转后变为 (0, 0, 0) -> (0, 0, 0)
                # 中心点 (w/2, h/2, 0) -> (w/2, 0, h/2) after rotation
                # 我们希望这个中心点移动到 (radius, 0, label_z_offset)
                # 所以需要平移 (radius - world_width/2, 0, label_z_offset - world_height/2) ?
                # 让我们简化：直接平移到 (radius, 0, label_z_offset)，这会将标签的 *原点* 放在那里
                # 为了将标签 *底部中心* 放在 (radius, 0, label_z_offset):
                # 需要将标签原点移动到 (radius - world_width / 2, 0, label_z_offset)
                label_img_item.translate(radius - world_width / 2.0, 0, label_z_offset) # x 中心对齐半径, y=0, z=offset

                # 5. 添加到视图
                self.gl_view.addItem(label_img_item)

            except Exception as e:
                print(f"创建半径 {radius} 的标签时出错: {e}")
                import traceback
                traceback.print_exc()
    def add_distance_circles(self, max_radius, step, num_points=100):
        """在 XY 平面 (z=0) 添加同心圆作为距离参考。"""
        radii = np.arange(step, max_radius + step, step) # 从 step 开始，包含 max_radius
        circle_color = (0.5, 0.5, 0.5, 0.6) # 半透明灰色
        z_offset = 0.0 # 可以设置一个微小的 Z 偏移量，以避免 Z-fighting (如果需要)

        for radius in radii:
            theta = np.linspace(0, 2 * np.pi, num_points) # 0 到 2*pi 的角度
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = np.full(num_points, z_offset) # 所有点都在 z=z_offset 平面

            circle_points = np.vstack([x, y, z]).transpose() # (N, 3) 形状的数组

            # 创建线图元来绘制圆圈
            circle_line = gl.GLLinePlotItem(pos=circle_points,
                                            color=circle_color,
                                            width=1, # 线宽
                                            mode='line_strip') # 连接所有点形成闭合路径
            self.gl_view.addItem(circle_line)

    @QtCore.pyqtSlot(object)
    def update_pointcloud_display(self, points):
        """更新 3D 点云视图。"""
        if not isinstance(points, np.ndarray) or points.size == 0:
            # 如果传入的不是有效的 numpy 数组，则返回
            fps = 0
            self.fps_label.setText(f"点云 FPS: {fps}")

            return
        try:
            # --- 应用变换 (可选) ---
            # 示例：如果需要，旋转点云
            if self.process_sensor:
                # 使点云坐标系的地面与世界坐标系的地面平行
                # 绕x轴旋转-45°
                # points = utils.rotate_points_around_x_axis(points, -45)
                ground_cloud, non_ground_cloud, ground_inlier_indices = utils.extract_ground_RANSAC_NormalVectorConstraint(points , distance_threshold=1.5)
    
                # --- 点云着色 ---
                # 简单的基于高度 (Z轴) 的着色
                z_coords = points[:, 2]
                min_z, max_z = np.min(z_coords), np.max(z_coords)
                if max_z > min_z: # 避免除以零
                    # 将 Z 值归一化到 [0, 1] 区间
                    normalized_z = (z_coords - min_z) / (max_z - min_z)
                    # 使用 pyqtgraph 的 colormap
                    cmap = pg.colormap.get('viridis') # 选项: 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'jet' 等
                    colors = cmap.map(normalized_z, mode='float') # 获取 RGBA 颜色 (N x 4 数组)

                else:
                    # 如果所有点高度相同，使用默认颜色
                    colors = np.array([[1.0, 1.0, 1.0, 1.0]] * len(points)) # 半透明白色

                colors[ground_inlier_indices] = np.array([[1.0, 0, 0, 0.7]] * len(ground_inlier_indices))
                # --- 更新绘图 ---
                self.point_cloud.setData(pos=points, color=colors, size=4, pxMode=True) # 更新点云的位置和颜色
            else: 
                # 绕x轴旋转-45°
                # points = utils.rotate_points_around_x_axis(points, -45)

                # --- 点云着色 ---
                # 简单的基于高度 (Z轴) 的着色
                z_coords = points[:, 2]
                min_z, max_z = np.min(z_coords), np.max(z_coords)
                if max_z > min_z: # 避免除以零
                    # 将 Z 值归一化到 [0, 1] 区间
                    normalized_z = (z_coords - min_z) / (max_z - min_z)
                    # 使用 pyqtgraph 的 colormap
                    cmap = pg.colormap.get('viridis') # 选项: 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'jet' 等
                    colors = cmap.map(normalized_z, mode='float') # 获取 RGBA 颜色 (N x 4 数组)
                    colors[:, 3] = 1.0
                else:
                    # 如果所有点高度相同，使用默认颜色
                    colors = np.array([[1.0, 1.0, 1.0, 1.0]] * len(points)) # 半透明白色

                # --- 更新绘图 ---
                # self.point_cloud.setData(pos=points, color=colors) # 更新点云的位置和颜色
                self.point_cloud.setGLOptions('opaque')
                self.point_cloud.setData(pos=points, color=colors, size=4, pxMode=True)  # 更新点云的位置和颜色
                
            # 可选：调整背景颜色
            # self.gl_view.setBackgroundColor('k') # 黑色背景

            # --- 更新点云 FPS 计数器 ---
            self.pointcloud_count += 1
            current_time = time.time()
            elapsed = current_time - self.last_pc_time
            if elapsed >= 1.0: # 每秒更新一次 FPS 显示
                fps = math.ceil(self.pointcloud_count / elapsed)
                self.fps_label.setText(f"点云 FPS: {fps}")
                self.pointcloud_count = 0 # 重置计数器
                self.last_pc_time = current_time # 更新时间戳

        except Exception as e:
            # 记录点云显示更新中的错误
            rospy.logerr(f"点云显示更新错误: {e}", exc_info=True)

    @QtCore.pyqtSlot(object)
    def update_pointcloud_display2(self, points, colors, info):
        """更新 3D 点云视图。"""
        if not isinstance(points, np.ndarray) or points.size == 0:
            # 如果传入的不是有效的 numpy 数组，则返回
            fps = 0
            self.fps_label.setText(f"点云 FPS: {fps}")

            return
        try:
            # --- 更新绘图 ---
            self.point_cloud.setGLOptions('opaque')
            self.point_cloud.setData(pos=points, color=colors, size=4, pxMode=True)  # 更新点云的位置和颜色
            # 可选：调整背景颜色
            # self.gl_view.setBackgroundColor('k') # 黑色背景
            self.result_text.appendPlainText(" ")
            self.result_text.appendPlainText(f"***** {self.ros_sensor_node.frame_count} *****")

            if hasattr(self, 'obb_lines'):
                for line_item in self.obb_lines:
                    self.gl_view.removeItem(line_item)

            if hasattr(self, 'obb_labels'):
                for label_item in self.obb_labels:
                    self.gl_view.removeItem(label_item)

            self.obb_lines = []  # 包围框列表
            self.obb_labels = []
            label_pos_text = ""
            for i in range(len(info)):
                bbox = info[i]
                # center = bbox[0]
                # dimensions = bbox[1]
                center = bbox['center']
                dimensions = bbox['extent']
                orientation = bbox['rotation']
                center_str = np.array2string(center, formatter={'float_kind': lambda x: f"{x:.2f}"})
                dimensions_str = np.array2string(dimensions, formatter={'float_kind': lambda x: f"{x:.2f}"})

                self.result_text.appendPlainText(f"障碍物 {i} : 位置 {center_str}   尺寸 {dimensions_str}")
                # 自动滚动到底部
                self.result_text.verticalScrollBar().setValue(self.result_text.verticalScrollBar().maximum())


                #----画包围框---#
                # 1. 计算顶点
                vertices = self.get_obb_vertices(center, dimensions, orientation)
                # 2. 生成线条数据
                pos, color = self.create_obb_lines(vertices)
                # 3. 创建 GLLinePlotItem 并添加到视图中
                line_item = gl.GLLinePlotItem(pos=pos, color=color, width=4.0, mode='lines')
                self.gl_view.addItem(line_item)
                self.obb_lines.append(line_item)

                # #---画标签----#
                label_pos_text = label_pos_text + f"障碍物 {i}: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})\n"
                label_text = f"{i}"
                img_data = render_text_to_image(label_text, size=20, text_color=(255, 255, 255, 255))
                if img_data is None or img_data.size == 0:
                    continue

                label_img_item = gl.GLImageItem(img_data, glOptions='translucent')

                label_pos = center + np.array([0, 0, dimensions[2] / 2 + 0.3])
                label_img_item.resetTransform()

                label_img_item.scale(-0.03, 0.03, 1.0, local=False)
                label_img_item.translate(label_pos[0], label_pos[1], label_pos[2])
                label_img_item.rotate(180, 0, 0, 1, local=True)


                self.gl_view.addItem(label_img_item)
                self.obb_labels.append(label_img_item)

            #更新障碍物位置坐标
            self.coord_label.setText(label_pos_text)
            self.coord_label.adjustSize()
            # --- 更新点云 FPS 计数器 ---
            self.pointcloud_count += 1
            current_time = time.time()
            elapsed = current_time - self.last_pc_time
            if elapsed >= 1.0:  # 每秒更新一次 FPS 显示
                fps = math.ceil(self.pointcloud_count / elapsed)
                self.fps_label.setText(f"点云 FPS: {fps}")
                self.pointcloud_count = 0  # 重置计数器
                self.last_pc_time = current_time  # 更新时间戳

        except Exception as e:
            # 记录点云显示更新中的错误
            rospy.logerr(f"点云显示更新错误: {e}", exc_info=True)

    @QtCore.pyqtSlot() # 标记为槽函数
    def save_single_frame(self):
        """手动保存当前相机帧。"""
        if not self.camera_ctl:
            self.append_log("错误: 相机未初始化。")
            return
        # 获取当前时间
        now = datetime.datetime.now()
        # 将当前时间转换为年月日时分秒毫秒格式
        current_date = now.strftime("%Y%m%d%H")
        timestamp = now.strftime("%Y%m%d%H%M%S") + f"{now.microsecond // 1000:03d}"  # 毫秒部分补零到3位
        frame, _ = self.camera_ctl.get_latest_frame()
        points = self.ros_sensor_node.get_latest_frame()

        if frame is not None:
            # 确定保存路径 - 需要一个运行中的会话或默认位置
            save_dir = self.data_saver.session_path
            if not save_dir or not self.data_saver.running:
                # 如果未在录制，可以保存到默认的 'snapshots' 文件夹
                save_dir = os.path.join(SAVE_PATH,'snapshots', current_date)
                os.makedirs(save_dir, exist_ok=True)
                    
            img_filename = f"{timestamp}.jpg"
            pcd_filename = f"{timestamp}.ply"
            os.makedirs( os.path.join(save_dir, 'image'), exist_ok=True)
            os.makedirs( os.path.join(save_dir, 'point_cloud'), exist_ok=True)
            img_full_path = os.path.join(save_dir, 'image',img_filename)
            pcd_full_path = os.path.join(save_dir, 'point_cloud',pcd_filename)

            try:
                # 确保 points 是 NumPy 数组且为 C-连续的 float32 类型
                if not isinstance(points, np.ndarray) or points.dtype != np.float32:
                    points = np.asarray(points, dtype=np.float32)
                if not points.flags['C_CONTIGUOUS']:
                    points = np.ascontiguousarray(points)

                # 检查点云数据的维度是否正确 (N x 3)
                if points.ndim != 2 or points.shape[1] != 3:
                    rospy.logerr(f"用于 PLY 保存的点云形状无效: {points.shape}")
                    return
                # 动态生成文件头
                header = _PLY_HEADER_TEMPLATE.format(vertex_count=points.shape[0])
                header_buf = header.encode('ascii') # 编码为 ASCII 字节

                # 使用带缓冲的文件写入 (64KB 缓冲区)
                with open(pcd_full_path, 'wb', buffering=65536) as f:
                    f.write(header_buf) # 写入文件头
                    f.write(points.tobytes()) # 使用 tobytes() 直接写入二进制数据
                self.append_log(f"点云已保存为: {pcd_full_path}") 
            except Exception as e:
                rospy.logerr(f"PLY 文件 {pcd_full_path} 保存失败: {e}")
                # 可以选择重新引发异常（如果是关键错误），或者只记录并继续
                # raise
            try:
                success = cv2.imwrite(img_full_path, frame)
                if success:
                    self.append_log(f"图像已保存为: {img_full_path}") 
                else:
                    self.append_log(f"错误: 无法保存单帧图像至 {img_full_path}")
            except Exception as e:
                self.append_log(f"错误: 保存单帧图像时出错: {e}")
                rospy.logerr(f"保存快照时出错: {e}")
        else:
            self.append_log("错误: 无法获取当前相机帧。")

    # 槽函数，连接到录制按钮的 toggled(bool) 信号
    @QtCore.pyqtSlot(bool)
    def toggle_recording(self, checked):
        """根据按钮状态开始或停止数据记录。"""
        if checked: # 按钮被按下 -> 开始录制
            try:
                # 检查所有必要的组件是否可用
                if not self.data_saver:
                    self.append_log("错误：DataSaver 未初始化。")
                    self.btn_record.setChecked(False) # 恢复按钮状态
                    return
                if not self.ros_sensor_node or not self.node_thread.is_alive():
                     self.append_log("错误：ROS 节点未运行。")
                     self.btn_record.setChecked(False) # 恢复按钮状态
                     return

                self.data_saver.start() # 尝试开始录制（此方法会发出日志消息）
                if self.data_saver.running: # 检查 start() 是否成功
                    self.recording = True
                    self.btn_record.setText("■ 停止记录")
                    # 设置属性以便 CSS 选择器可以根据状态应用样式
                    self.btn_record.setProperty("recording", True)
                else:
                     # 开始失败（例如，无法创建目录），恢复按钮状态
                     self.recording = False
                     self.btn_record.setChecked(False)
                     self.btn_record.setText("● 开始记录")
                     self.btn_record.setProperty("recording", False)
            except Exception as e:
                self.append_log(f"启动记录时出错: {e}")
                rospy.logerr(f"启动记录时出错: {e}", exc_info=True)
                # 出错时确保状态和按钮一致
                self.recording = False
                self.btn_record.setChecked(False)
                self.btn_record.setText("● 开始记录")
                self.btn_record.setProperty("recording", False)
        else: # 按钮被释放 -> 停止录制
            try:
                if self.data_saver:
                    self.data_saver.stop() # 尝试停止录制（此方法会发出日志消息）
                self.recording = False
                self.btn_record.setText("● 开始记录")
                self.btn_record.setProperty("recording", False) # 更新属性
            except Exception as e:
                 self.append_log(f"停止记录时出错: {e}")
                 rospy.logerr(f"停止记录时出错: {e}", exc_info=True)
                 # 即使停止出错，也要保持 UI 状态一致
                 self.recording = False
                 self.btn_record.setText("● 开始记录")
                 self.btn_record.setProperty("recording", False)

        # 刷新按钮样式以应用基于属性的 CSS 规则
        self.btn_record.style().unpolish(self.btn_record)
        self.btn_record.style().polish(self.btn_record)
 

    def process_sensor_data(self):
        """用于自定义点云处理的占位符函数。"""
        if self.process_sensor :
            self.process_sensor = False
        else:
            self.process_sensor = True
        # self.append_log("点云数据处理功能 (未实现)...")
        # 在此处添加实际的处理逻辑

    def on_checkBox_process_cloud_clicked(self):
        if self.checkBox_process_cloud.isChecked():
            self.process_sensor = True
        else:
            self.process_sensor = False

    def on_checkBox_obstacle_recog_clicked(self):
        #需要清理掉队列
        while not self.ros_sensor_node.data_queue.empty():
            self.ros_sensor_node.data_queue.get_nowait()
        while not self.input_obstacle_queue.empty():
            self.input_obstacle_queue.get_nowait()
        while not self.input_land_queue.empty():
            self.input_land_queue.get_nowait()

        while not self.obstacle_res_queue.empty():
            self.obstacle_res_queue.get_nowait()
        # 移除障碍物框
        if hasattr(self, 'obb_lines'):
            for line_item in self.obb_lines:
                self.gl_view.removeItem(line_item)
            self.obb_lines.clear()
        # 移除障碍物标签
        if hasattr(self, 'obb_labels'):
            for label_item in self.obb_labels:
                self.gl_view.removeItem(label_item)
            self.obb_labels.clear()
        self.coord_label.setText("")

        #只要障碍物检测状态发生变化 就关掉语义识别的复选框
        self.flag_semantics_evt.clear()
        self.checkBox_semantics.setChecked(False)

        if self.checkBox_obstacle_recog.isChecked():
            self.ros_sensor_node.flag_recog = True
            self.flag_recog_evt.set()
            #语义识别必须障碍物检测打开才能打开
            self.checkBox_semantics.setEnabled(True)

        else:
            self.ros_sensor_node.flag_recog = False
            self.flag_recog_evt.clear()
            self.checkBox_semantics.setEnabled(False)

            # while not self.obstacle_res_queue.empty():
            #     self.obstacle_res_queue.get_nowait()
            # #移除障碍物框
            # if hasattr(self, 'obb_lines'):
            #     for line_item in self.obb_lines:
            #         self.gl_view.removeItem(line_item)
            #     self.obb_lines.clear()
            # # 移除障碍物标签
            # if hasattr(self, 'obb_labels'):
            #     for label_item in self.obb_labels:
            #         self.gl_view.removeItem(label_item)
            #     self.obb_labels.clear()
            # self.coord_label.setText("")

    #语义识别标志位
    def on_checkBox_semantics_clicked(self):
        if self.checkBox_semantics.isChecked():
            self.flag_semantics_evt.set()
        else:
            self.flag_semantics_evt.clear()

    def on_checkBox_landing_recog_clicked(self):
        # 需要清理掉队列
        while not self.ros_sensor_node.data_queue.empty():
            self.ros_sensor_node.data_queue.get_nowait()
        while not self.input_land_queue.empty():
            self.input_land_queue.get_nowait()
        while not self.input_obstacle_queue.empty():
            self.input_obstacle_queue.get_nowait()
        # 移除旧的着陆区圆形
        if not self.landing_zone_circle_delete:
            if self.landing_zone_circle is not None:
                self.gl_view.removeItem(self.landing_zone_circle)
                self.landing_zone_circle_delete = True
        while not self.land_queue.empty():
            self.land_queue.get_nowait()

        if self.checkBox_landing.isChecked():
            self.ros_sensor_node.flag_landing = True
            self.flag_land_evt.set()
        else:
            self.ros_sensor_node.flag_landing = False
            self.flag_land_evt.clear()
            # #移除旧的着陆区圆形
            # if not self.landing_zone_circle_delete:
            #     if self.landing_zone_circle is not None:
            #         self.gl_view.removeItem(self.landing_zone_circle)
            #         self.landing_zone_circle_delete = True
            # while not self.land_queue.empty():
            #     self.land_queue.get_nowait()


    #画最小包围框的相关函数
    def get_obb_vertices(self,center, extent, orientation):
        """
        根据中心、尺寸和旋转矩阵计算OBB的8个顶点坐标。

        Args:
            center (np.ndarray): 包围框中心点 [x, y, z]
            extent (np.ndarray): 包围框在三个轴上的尺寸 [dx, dy, dz]
            orientation (np.ndarray): 3x3 旋转矩阵

        Returns:
            np.ndarray: 8x3 的数组，包含8个顶点的坐标
        """
        # 1. 计算局部坐标系下，从中心到各个面的半距离
        dx, dy, dz = extent / 2.0

        # 2. 定义8个顶点相对于中心的偏移量（在局部坐标系下）
        # 顺序可以任意，但通常按“下-上”的顺序定义
        local_offsets = np.array([
            [-dx, -dy, -dz],
            [dx, -dy, -dz],
            [dx, dy, -dz],
            [-dx, dy, -dz],
            [-dx, -dy, dz],
            [dx, -dy, dz],
            [dx, dy, dz],
            [-dx, dy, dz]
        ])

        # 3. 通过旋转矩阵将局部偏移量转换到世界坐标系
        # R @ local_offsets.T 是矩阵乘法，计算所有点的旋转
        rotated_offsets = (orientation @ local_offsets.T).T

        # 4. 加上中心点坐标，得到世界坐标系下的顶点位置
        world_vertices = center + rotated_offsets

        return world_vertices

    def get_aabb_vertices(self, center, extent):
        """
        根据中心和尺寸计算AABB的8个顶点坐标（无旋转）。

        Args:
            center (np.ndarray): 包围框中心点 [x, y, z]
            extent (np.ndarray): 包围框在三个轴上的尺寸 [dx, dy, dz]

        Returns:
            np.ndarray: 8x3 的数组，包含8个顶点的坐标
        """
        dx, dy, dz = extent / 2.0

        # AABB的顶点计算非常直接，就是中心点加减各个方向的半长
        vertices = np.array([
            center + [-dx, -dy, -dz],
            center + [dx, -dy, -dz],
            center + [dx, dy, -dz],
            center + [-dx, dy, -dz],
            center + [-dx, -dy, dz],
            center + [dx, -dy, dz],
            center + [dx, dy, dz],
            center + [-dx, dy, dz]
        ])

        return vertices
    def create_obb_lines(self,vertices):
        """
        根据OBB的8个顶点，创建用于绘制12条边的点坐标和连接方式。

        Args:
            vertices (np.ndarray): 8x3 的顶点坐标数组

        Returns:
            tuple: (pos, color) pos是Nx3的点数组，color是Nx4的颜色数组
        """
        # 定义12条边的连接方式（通过顶点索引）
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
            (0, 4), (1, 5), (2, 6), (3, 7)  # 垂直边
        ]

        # 生成用于绘制线条的点坐标
        line_points = []
        for start_idx, end_idx in edges:
            line_points.append(vertices[start_idx])
            line_points.append(vertices[end_idx])

        pos = np.array(line_points)

        # 为每条线设置颜色，例如：红色
        color = np.array([[1, 0, 0, 1] for _ in range(len(pos))])
        return pos, color

    def _create_text_image(self, text, color=(255, 255, 255), bg_color=(0, 0, 0, 0), font_size=12):
        """生成一张包含指定文字的QImage。"""
        font = QtGui.QFont('Arial', font_size)
        fm = QtGui.QFontMetrics(font)
        text_width = fm.width(text)
        text_height = fm.height()
        image = QtGui.QImage(text_width, text_height, QtGui.QImage.Format_ARGB32)
        image.fill(QtGui.QColor(*bg_color))
        painter = QtGui.QPainter(image)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setFont(font)
        painter.setPen(QtGui.QColor.fromRgb(*color))
        painter.drawText(image.rect(), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, text)
        painter.end()
        return image

    def check_land_results(self):
        start_time = time.time()
        if not hasattr(self, 'timer_land') or not self.timer_land.isActive():
            return
        if self.ros_sensor_node is None:
            return
         # 如果着陆检测没打开
        if not self.ros_sensor_node.flag_landing:
            # 界面还有圆环
            if not self.landing_zone_circle_delete:
                if self.landing_zone_circle is not None:
                    self.gl_view.removeItem(self.landing_zone_circle)
                    self.landing_zone_circle_delete = True
        #1、着陆检测和障碍物检测都未开启
        if not self.ros_sensor_node.flag_landing and not self.ros_sensor_node.flag_recog:
            return


        #2、障碍物检测打开 着陆检测未打开
        if self.ros_sensor_node.flag_recog and not self.obstacle_res_queue.empty() and not self.ros_sensor_node.flag_landing:
            obstacle_res = self.obstacle_res_queue.get_nowait()
            self.update_pointcloud_display2(obstacle_res[1],obstacle_res[2],obstacle_res[3])

        #3、障碍物检测未打开 着陆检测打开
        if self.ros_sensor_node.flag_landing and not self.land_queue.empty() and not self.ros_sensor_node.flag_recog:
            land_zone = self.land_queue.get_nowait()
            if land_zone[1] is not None:
                x = land_zone[1]['center_x']
                y = land_zone[1]['center_y']
                z = land_zone[1]['center_z']
                points = land_zone[2]
                self.update_landing_zone(x,y,z)
                self.update_pointcloud_display(points[::5])
            else:
                # print("未检测到合适降落区")
                #未检测到着陆区，也显示原始点云,但是要把之前的着陆区圆环删除
                if not self.landing_zone_circle_delete:
                    if self.landing_zone_circle is not None:
                        self.gl_view.removeItem(self.landing_zone_circle)
                        self.landing_zone_circle_delete = True
                #点云显示
                points = land_zone[2]
                self.update_pointcloud_display(points[::5])

        # 4、障碍物检测打开 着陆检测打开 需要匹配帧号
        if self.ros_sensor_node.flag_recog and self.ros_sensor_node.flag_landing:
            # --- 1. 从着陆区队列获取结果 ---
            # --- 1. 清空着陆区队列，将所有新结果放入缓存 ---

            while True:
                try:
                    land_frame_id, land_zone, points_array = self.land_queue.get_nowait()
                    if land_frame_id not in self.result_cache:
                        self.result_cache[land_frame_id] = {}
                    self.result_cache[land_frame_id]['land'] = (land_zone, points_array)
                    # print(f"缓存着陆区帧ID: {land_frame_id}")
                except:
                    # 队列已空，跳出循环
                    break

            # --- 2. 从障碍物队列获取结果 ---
            try:
                # 工作进程现在返回 (frame_id, merged_points, merged_colors, cluster_infos)
                obs_frame_id, merged_points, merged_colors, cluster_infos = self.obstacle_res_queue.get_nowait()

                # 立即显示障碍物结果
                self.update_pointcloud_display2(merged_points, merged_colors, cluster_infos)

                # --- 3. 搜寻并等待匹配的着陆区结果 ---
                land_result_to_use = None
                matched_land_frame_id = None

                wait_start_time = time.time()
                max_wait_time = 0.1  # 等待窗口：20毫秒
                # match_success = False
                while time.time() - wait_start_time < max_wait_time:
                    # --- 步骤 A: 搜寻 ---
                    # 直接用当前障碍物帧号去缓存里搜寻
                    if obs_frame_id in self.result_cache and 'land' in self.result_cache[obs_frame_id]:
                        land_result_to_use = self.result_cache[obs_frame_id]['land']
                        matched_land_frame_id = obs_frame_id
                        # print(f"成功匹配！找到同帧着陆区结果，帧ID: {obs_frame_id}")
                        # match_success = True
                        break  # 找到了，跳出等待循环

                    # --- 步骤 B: 如果没找到，就等一下 ---
                    # 在等待期间，不断从队列里取数据，更新缓存，为下一次搜寻做准备
                    while True:
                        try:
                            land_frame_id, land_zone, points_array = self.land_queue.get_nowait()
                            if land_frame_id not in self.result_cache:
                                self.result_cache[land_frame_id] = {}
                            self.result_cache[land_frame_id]['land'] = (land_zone, points_array)
                        except:
                            # 队列已空，跳出循环
                            break

                # --- 4. 根据匹配结果更新着陆区显示 ---
                if land_result_to_use and land_result_to_use[0] is not None:
                    x = land_result_to_use[0]['center_x']
                    y = land_result_to_use[0]['center_y']
                    z = land_result_to_use[0]['center_z']
                    self.update_landing_zone(x, y, z)
                    self.landing_zone_circle_delete = False
                else:
                    if not land_result_to_use:
                        print(f"等待超时，未找到帧ID {obs_frame_id} 对应的着陆区结果。")
                    if not self.landing_zone_circle_delete:
                        if self.landing_zone_circle is not None:
                            self.gl_view.removeItem(self.landing_zone_circle)
                            self.landing_zone_circle_delete = True

                # --- 5. 清理已处理和过时的数据 ---
                # a. 清理已使用的着陆区结果
                if matched_land_frame_id is not None and matched_land_frame_id in self.result_cache:
                    del self.result_cache[matched_land_frame_id]

                # b. 清理所有比当前障碍物帧更旧的着陆区结果（它们已经没用了）
                old_frame_ids = [fid for fid in self.result_cache.keys() if fid < obs_frame_id]
                for fid in old_frame_ids:
                    print(f"清理过时的着陆区缓存，帧ID: {fid}")
                    del self.result_cache[fid]

            except:

                pass

                # --- 6. (可选) 防止缓存无限增长 ---
            max_frames_to_keep = 1000
            if len(self.result_cache) > max_frames_to_keep:
                try:
                    oldest_frame_id = next(iter(self.result_cache))
                    print(f"警告: 着陆区缓存过大，强制丢弃旧帧 {oldest_frame_id}。")
                    del self.result_cache[oldest_frame_id]
                except StopIteration:
                    pass

            end_time = time.time()
            elapsed_time_ms = (end_time - start_time) * 1000

            target_duration_ms = 80.0  # 目标运行时间（毫秒）
            remaining_time_ms = target_duration_ms - elapsed_time_ms

            # 只有当实际运行时间小于目标时间时才需要等待
            if remaining_time_ms > 0:
                # print(f"算法运行了 {elapsed_time_ms:.2f} ms, 开始精确等待 {remaining_time_ms:.2f} ms")
                wait_start = time.perf_counter()  # 使用更高精度的计时器

                # 忙等待循环
                while (time.perf_counter() - wait_start) * 1000 < remaining_time_ms:
                    time.sleep(0.001)  # 等待1毫秒

    def check_land_results0(self):
        start_time = time.time()
        if not hasattr(self, 'timer_land') or not self.timer_land.isActive():
            return
        if self.ros_sensor_node is None:
            return
         # 如果着陆检测没打开
        if not self.ros_sensor_node.flag_landing:
            # 界面还有圆环
            if not self.landing_zone_circle_delete:
                if self.landing_zone_circle is not None:
                    self.gl_view.removeItem(self.landing_zone_circle)
                    self.landing_zone_circle_delete = True
        #1、着陆检测和障碍物检测都未开启
        if not self.ros_sensor_node.flag_landing and not self.ros_sensor_node.flag_recog:
            return

        #2、障碍物检测打开 着陆检测未打开
        if self.ros_sensor_node.flag_recog and not self.obstacle_res_queue.empty() and not self.ros_sensor_node.flag_landing:
            obstacle_res = self.obstacle_res_queue.get_nowait()
            self.update_pointcloud_display2(obstacle_res[1],obstacle_res[2],obstacle_res[3])

        #3、障碍物检测未打开 着陆检测打开
        if self.ros_sensor_node.flag_landing and not self.land_queue.empty() and not self.ros_sensor_node.flag_recog:
            land_zone = self.land_queue.get_nowait()
            if land_zone[1] is not None:
                x = land_zone[1]['center_x']
                y = land_zone[1]['center_y']
                z = land_zone[1]['center_z']
                points = land_zone[2]
                self.update_landing_zone(x,y,z)
                self.update_pointcloud_display(points[::5])
            else:
                print("未检测到合适降落区")
                #未检测到着陆区，也显示原始点云,但是要把之前的着陆区圆环删除
                if not self.landing_zone_circle_delete:
                    if self.landing_zone_circle is not None:
                        self.gl_view.removeItem(self.landing_zone_circle)
                        self.landing_zone_circle_delete = True
                #点云显示
                points = land_zone[2]
                self.update_pointcloud_display(points[::5])

        # 4、障碍物检测打开 着陆检测打开 需要匹配帧号
        if self.ros_sensor_node.flag_recog and self.ros_sensor_node.flag_landing:
            while not self.land_queue.empty():
                try:
                    land_frame_id, land_zone, points_array = self.land_queue.get_nowait()
                    if land_frame_id not in self.result_cache:
                        self.result_cache[land_frame_id] = {}
                    self.result_cache[land_frame_id]['land'] = (land_zone, points_array)
                except queue.Empty:
                    # 队列已空，跳出循环
                    break
            while not self.obstacle_res_queue.empty():
                try:
                    frame_id, merged_points, merged_colors, clustered_informations = self.obstacle_res_queue.get_nowait()
                    if frame_id not in self.result_cache:
                        self.result_cache[frame_id] = {}
                    self.result_cache[frame_id]['obstacle'] = (merged_points, merged_colors, clustered_informations)
                except queue.Empty:
                    break

            if not self.result_cache:
                return

            # 假设ID从0到999循环
            MAX_FRAME_ID = 999
            expected_next_id = (self.last_processed_frame_id + 1) % (MAX_FRAME_ID + 1)
            if expected_next_id in self.result_cache:
                cache_entry = self.result_cache[expected_next_id]
                land_ready = 'land' in cache_entry
                obstacle_ready = 'obstacle' in cache_entry
                print("ssssssssssssss")

                if land_ready and obstacle_ready:
                    self.update_pointcloud_display2(cache_entry['obstacle'][0], cache_entry['obstacle'][1],
                                                    cache_entry['obstacle'][2])

                    if cache_entry['land'][0] is not None:
                        x = cache_entry['land'][0]['center_x']
                        y = cache_entry['land'][0]['center_y']
                        z = cache_entry['land'][0]['center_z']
                        self.update_landing_zone(x, y, z)
                        self.landing_zone_circle_delete = False
                    else:
                        if not self.landing_zone_circle_delete:
                            if self.landing_zone_circle is not None:
                                self.gl_view.removeItem(self.landing_zone_circle)
                                self.landing_zone_circle_delete = True

                    del self.result_cache[expected_next_id]
                    self.last_processed_frame_id = expected_next_id
        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000

        target_duration_ms = 80.0  # 目标运行时间（毫秒）
        remaining_time_ms = target_duration_ms - elapsed_time_ms

        # 只有当实际运行时间小于目标时间时才需要等待
        if remaining_time_ms > 0:
            # print(f"算法运行了 {elapsed_time_ms:.2f} ms, 开始精确等待 {remaining_time_ms:.2f} ms")
            wait_start = time.perf_counter()  # 使用更高精度的计时器

            # 忙等待循环
            while (time.perf_counter() - wait_start) * 1000 < remaining_time_ms:
                time.sleep(0.001)  # 等待1毫秒

    def closeEvent(self, event):
        """处理窗口关闭事件，确保资源被正确释放。"""
        self.append_log("正在关闭应用程序...")

        # 1. 停止录制
        if self.data_saver and self.data_saver.running:
            self.append_log("正在停止数据记录...")
            self.data_saver.stop() # 允许保存器完成写入

        # 2. 停止 ROS 节点和传感器轮询
        if hasattr(self, 'ros_sensor_node') and self.ros_sensor_node:
            self.ros_sensor_node.stop() # 停止节点内部循环

        # 3. 停止相机捕获线程
        if self.camera_ctl:
            self.camera_ctl.stop() # 停止相机线程并释放资源

        # 4. 等待线程结束（可选，守护线程通常会自动退出）
        if hasattr(self, 'node_thread') and self.node_thread.is_alive():
            self.node_thread.join(timeout=1) # 等待 ROS 节点线程结束

        # 5. 释放其他资源（例如，如果读取器需要手动关闭连接）
        if self.imu_reader and hasattr(self.imu_reader, 'close'):
            self.imu_reader.close()
        if self.gps_reader and hasattr(self.gps_reader, 'close'):
            self.gps_reader.close()



        # 首先停止定时器，避免继续处理队列
        if hasattr(self, 'timer_land') and self.timer_land.isActive():
            self.timer_land.stop()
        #关闭数据分发进程

        # 清空队列，避免阻塞
        while not self.input_obstacle_queue.empty():
            try:
                self.input_obstacle_queue.get_nowait()
            except:
                pass

        while not self.input_land_queue.empty():
            try:
                self.input_land_queue.get_nowait()
            except:
                pass

            # 设置停止事件，通知工作进程退出
        self.data_stop_evt.set()
        # 等待一小会儿，让进程有机会退出
        time.sleep(0.5)
        if self.data_proc and self.data_proc.is_alive():
            print("正在停止数据分发进程...")

            self.data_proc.join(timeout=5.0)  # 最多等待2秒
            # 如果进程仍未结束，强制终止
            if self.data_proc.is_alive():
                print("进程未正常结束，强制终止...")
                self.data_proc.terminate()
                self.data_proc.join(timeout=1.0)
                if self.data_proc.is_alive():
                    self.data_proc.kill()  # Python 3.7+
                    self.data_proc.join()
        #关闭着陆区进程
        # 清空队列，避免阻塞
        while not self.land_queue.empty():
            try:
                self.land_queue.get_nowait()
            except:
                pass
            # 设置停止事件，通知工作进程退出
        self.stop_evt.set()
        # 等待一小会儿，让进程有机会退出
        time.sleep(0.5)
        # print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!{self.proc.is_alive()}")
        if self.proc and self.proc.is_alive():
            print("正在停止着陆检测子进程...")
            # 设置停止事件
            # self.stop_evt.set()
            # 等待进程结束
            self.proc.join(timeout=5.0)  # 最多等待2秒
            # 如果进程仍未结束，强制终止
            if self.proc.is_alive():
                print("进程未正常结束，强制终止...")
                self.proc.terminate()
                self.proc.join(timeout=1.0)
                if self.proc.is_alive():
                    self.proc.kill()  # Python 3.7+
                    self.proc.join()

        #关闭障碍物检测进程
            # 清空队列，避免阻塞
            while not self.obstacle_res_queue.empty():
                try:
                    self.obstacle_res_queue.get_nowait()
                except:
                    pass
                # 设置停止事件，通知工作进程退出
            self.obstacle_stop_evt.set()
            # 等待一小会儿，让进程有机会退出
            time.sleep(0.5)

            if self.obstacle_proc and self.obstacle_proc.is_alive():
                print("正在停止障碍物检测子进程...")
                # 设置停止事件
                # self.stop_evt.set()
                # 等待进程结束
                self.obstacle_proc.join(timeout=5.0)  # 最多等待2秒
                # 如果进程仍未结束，强制终止
                if self.obstacle_proc.is_alive():
                    print("进程未正常结束，强制终止...")
                    self.obstacle_proc.terminate()
                    self.obstacle_proc.join(timeout=1.0)
                    if self.obstacle_proc.is_alive():
                        self.obstacle_proc.kill()  # Python 3.7+
                        self.obstacle_proc.join()

        # 通知 ROS 关闭，这样 rospy.spin() 会退出
        rospy.signal_shutdown("GUI 已关闭")
        self.append_log("清理完成，正在退出。")
        super().closeEvent(event) # 调用父类的 closeEvent

    @staticmethod
    def worker(pcd_q, land_queue: Queue, stop_evt: Event):
        from LandRecog2 import LandingZoneDetector

        detector = LandingZoneDetector()
        # 防止队列阻塞进程退出
        land_queue.cancel_join_thread()

        while not stop_evt.is_set():

            # start_time = time.time()
            # 确保获取到的是最新的一帧
            point = o3d.geometry.PointCloud()

            if not pcd_q.empty():
                # points_array = pcd_q.get_nowait()
                try:
                    frame_id, points_array = pcd_q.get(timeout=0.1)  # 非阻塞 + 超时
                except:
                    continue

                if points_array is not None and len(points_array) > 0:
                    # print(f"盲降获取到数据，点数: {len(points_array)}")
                    # 创建PointCloud对象
                    point.points = o3d.utility.Vector3dVector(points_array)
                    detector.spinOnce(point, stop_evt)
                    land_zone = detector.best_landing_zone

                    # 将结果放入队列
                    #if land_zone is not None:
                        # land_queue.put(land_zone)

                    land_queue.put((frame_id,land_zone,points_array)) #把点云和检测结果一起传回显示，保证同步


                    # end_time = time.time()
                    # elapsed_time = end_time - start_time
                    # print(f"盲降代码运行时间: {elapsed_time:.4f} 秒")
                else:
                    continue

            else:
                time.sleep(0.1)
                # end_time = time.time()
                # elapsed_time = end_time - start_time
                # print(f"盲降代码运行时间: {elapsed_time:.4f} 秒")
                continue
        # detector.run()

    @staticmethod
    def obstacle_worker(pcd_q, obstacle_res_queue: Queue, obstacle_stop_evt: Event, semantics_evt: Event):
        from ObstacleRecog import ObstacleRecog
        # print("障碍物进程")
        obstacle_detector = ObstacleRecog()
        # 防止队列阻塞进程退出
        obstacle_res_queue.cancel_join_thread()

        while not obstacle_stop_evt.is_set():

            # start_time = time.time()
            # 确保获取到的是最新的一帧
            point = o3d.geometry.PointCloud()

            if not pcd_q.empty():
                # points_array = pcd_q.get_nowait()
                try:
                    frame_id, points_array = pcd_q.get(timeout=0.1)  # 非阻塞 + 超时
                    # print(f"障碍物获取到数据，点数: {len(points_array)}")
                except:
                    continue

                if points_array is not None and len(points_array) > 0:

                    # 创建PointCloud对象
                    point.points = o3d.utility.Vector3dVector(points_array)
                    if semantics_evt.is_set():
                        obstacle_detector.spinOnce(point, obstacle_stop_evt)
                    else:
                        obstacle_detector.spinOnce_no_semantic(point, obstacle_stop_evt)


                    merged_points_copy = copy.deepcopy(obstacle_detector.merged_points)
                    merged_colors_copy = copy.deepcopy(obstacle_detector.merged_colors)
                    cluster_infos_copy = copy.deepcopy(obstacle_detector.clustered_informations)

                    # 将副本放入队列
                    obstacle_res_queue.put((frame_id, merged_points_copy, merged_colors_copy, cluster_infos_copy))
                    # obstacle_res_queue.put((frame_id, obstacle_detector.merged_points,obstacle_detector.merged_colors,obstacle_detector.clustered_informations))

                else:
                    continue

            else:
                time.sleep(0.1)
                continue

    # --- 分发器进程 ---
    @staticmethod
    def data_dispatcher(main_queue: Queue, landing_queue: Queue, obstacle_queue: Queue, stop_evt: Event, recog_evt: Event, land_evt: Event, shared_frame_id: Value):
        print("[分发器] 进程已启动。")
        obstacle_queue.cancel_join_thread()
        landing_queue.cancel_join_thread()
        while not stop_evt.is_set():
            try:
                points = main_queue.get(timeout=0.5)
                # 障碍物检测需要下采样
                points_downsampled = points[::5]

                with shared_frame_id.get_lock():
                    current_id = shared_frame_id.value
                    shared_frame_id.value = shared_frame_id.value + 1
                    # shared_frame_id.value = (shared_frame_id.value + 1) % 1000

                # 将数据同步放入两个工作队列
                if recog_evt.is_set() and not land_evt.is_set():
                    if obstacle_queue.qsize() > 10:  # 队列累积10清空
                        while not obstacle_queue.empty():
                            obstacle_queue.get_nowait()
                    obstacle_queue.put((current_id ,points_downsampled))
                if land_evt.is_set() and not recog_evt.is_set():
                    if landing_queue.qsize() > 10:  # 队列累积10清空
                        while not landing_queue.empty():
                            landing_queue.get_nowait()
                    landing_queue.put((current_id, points))

                if land_evt.is_set() and recog_evt.is_set():
                    if landing_queue.qsize() > 10 or landing_queue.qsize() > 10:  # 队列累积10清空
                        while not landing_queue.empty():
                            landing_queue.get_nowait()
                        while not obstacle_queue.empty():
                            obstacle_queue.get_nowait()
                    landing_queue.put_nowait((current_id, points))
                    obstacle_queue.put_nowait((current_id, points_downsampled))
            except:
                continue  # 超时或队列为空，继续循环


    def start_process(self):
        #分发器进程
        self.data_proc = Process(target=self.data_dispatcher,
                                 args=(self.ros_sensor_node.data_queue, self.input_land_queue, self.input_obstacle_queue, self.data_stop_evt, self.flag_recog_evt, self.flag_land_evt, self.shared_frame_id))
        self.data_proc.start()
        #着陆区检测子进程
        """
            input_land_queue---[points,..,]
            land_queue---[（frame_id,land_zone,points）,...,]
        """
        self.proc = Process(target=self.worker, args=(self.input_land_queue, self.land_queue, self.stop_evt))
        self.proc.start()
        #障碍物检测子进程
        """
            input_obstacle_queue---[points,..,]
            obstacle_res_queue---[（frame_id, merged_points,merged_colors,clustered_informations）,...,]
        """
        self.obstacle_proc = Process(target=self.obstacle_worker,
                                     args=(self.input_obstacle_queue, self.obstacle_res_queue, self.obstacle_stop_evt, self.flag_semantics_evt))
        self.obstacle_proc.start()


    def create_circle_outline0(self, center_x, center_y, center_z, radius=8):
        """
        创建一个绿色圆圈轮廓
        """
        # 生成圆上的点
        theta = np.linspace(0, 2 * np.pi, 100)
        x = radius * np.cos(theta) + center_x
        y = radius * np.sin(theta) + center_y
        z = np.full_like(theta, center_z)

        # 创建点数组
        points = np.vstack([x, y, z]).T

        # 创建红色圆圈
        circle = gl.GLLinePlotItem(
            pos=points,
            color=(0, 1, 0, 1),  # 红色，完全不透明
            width=3,  # 线宽
            mode='line_strip'  # 连接所有点形成闭合环
        )

        return circle

    def create_circle_outline(self, center_x, center_y, center_z, radius=8, color=(0, 1, 0, 0.8)):
        """
        创建一个圆盘（三角网格）

        Args:
            center_x, center_y, center_z: 圆盘中心坐标
            radius: 半径
            color: 颜色 (R, G, B, A)，默认绿色半透明
        """
        # 创建圆盘的顶点
        sectors = 64  # 扇形数量，越大越圆滑
        vertices = []
        faces = []

        # 中心点
        vertices.append([center_x, center_y, center_z])

        # 圆周上的点
        theta = np.linspace(0, 2 * np.pi, sectors, endpoint=False)
        for t in theta:
            x = radius * np.cos(t) + center_x
            y = radius * np.sin(t) + center_y
            z = center_z
            vertices.append([x, y, z])

        vertices = np.array(vertices)

        # 创建三角面（中心点 + 相邻两个圆周点）
        for i in range(1, sectors + 1):
            # 使用 modulo 实现闭合
            next_i = i % sectors + 1
            faces.append([0, i, next_i])

        faces = np.array(faces)

        # 创建网格对象
        disk = gl.GLMeshItem(
            vertexes=vertices,
            faces=faces,
            faceColors=np.tile(color, (len(faces), 1)),  # 每个面设置颜色
            smooth=False,
            drawEdges=False,  # 不画边框
            shader='balloon',  # 使用着色器实现更好的光照效果
            glOptions = 'additive'
        )

        return disk

    def update_landing_zone(self, x, y, z):
        """
        更新降落区圆圈的位置
        """
        # 移除现有的圆圈（如果存在）
        if not self.landing_zone_circle_delete:
            if self.landing_zone_circle is not None:
                self.gl_view.removeItem(self.landing_zone_circle)

        # 创建新的圆圈
        self.landing_zone_circle = self.create_circle_outline(x, y, z)

        # 添加到视图
        self.gl_view.addItem(self.landing_zone_circle)
        self.landing_zone_circle_delete = False


def setup_multiprocessing():
    """
    spawn 不会复制父进程的内存。它会启动一个全新的、干净的 Python 解释器进程。
    这个新进程会重新执行主模块的代码（从 import 语句开始），直到它被分配去执行 target 函数
    防止open3d RANSAC算法卡住
    """
    import multiprocessing as mp
    try:
        # 'spawn' 是最干净的方式，它会启动一个全新的Python解释器
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        # 在某些环境中，如果已经设置过，可能会抛出RuntimeError
        print("Multiprocessing start method already set.")
# --- 主程序入口 ---
if __name__ == '__main__':
    setup_multiprocessing()

    try:
        app = QApplication(sys.argv) # 创建 Qt 应用程序实例
        window = MainWindow(app) # 创建主窗口实例
        window.show() # 显示主窗口
        sys.exit(app.exec_()) # 进入 Qt 事件循环
    except Exception as e:
        # 捕获初始化期间的顶层错误
        logging.critical(f"启动应用程序失败: {e}", exc_info=True)
        # 如果 GUI 甚至没有启动，尝试显示一个简单的消息框
        # 检查 QApplication 是否已创建
        if 'app' not in locals() or QApplication.instance() is None:
             # 如果 Qt 本身严重失败，这部分可能也无法工作
             try:
                  # 尝试创建一个临时的 app 来显示消息
                  error_app = QApplication(sys.argv)
                  QMessageBox.critical(None, "启动错误", f"无法启动应用程序:\n{e}")
                  sys.exit(1)
             except Exception as e2:
                  # 如果连消息框都无法显示，打印到 stderr
                  print(f"严重错误: 启动应用程序失败: {e}\n显示消息时的次要错误: {e2}", file=sys.stderr)
                  sys.exit(1)
        else:
             # 如果 app 存在但窗口初始化失败
             QMessageBox.critical(None, "启动错误", f"无法初始化主窗口:\n{e}")
             sys.exit(1)