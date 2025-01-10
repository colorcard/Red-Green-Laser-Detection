# debug_laser_tracker.py
import cv2
import numpy as np
import time
from datetime import datetime
import threading
import serial
import serial.tools.list_ports
from MultThread_Main import LaserTracker  # 请将 "your_laser_tracker_file" 替换为实际保存 LaserTracker 类的文件名

class DebugLaserTracker(LaserTracker):
    """
    继承自 LaserTracker，用于调试时查看各类中间 mask。
    与 macOS 上的 GUI 线程限制兼容，建议在 main 线程中进行 imshow 操作。
    """

    def __init__(self):
        super().__init__()
        # 额外保存调试用的 mask
        self.black_mask_debug = None
        self.red_mask_debug = None
        self.green_mask_debug = None

    def process_frame(self, frame):
        """
        重写父类的 process_frame 方法：
        - 调用父类逻辑
        - 额外保存黑色、红色和绿色部分的遮罩 (mask) 到内部变量，便于在主线程中调试显示
        """
        start_process = time.time()
        frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)

        # 父类中的滑块值更新
        self.update_hsv_values()

        # 黑色外框 mask
        self.black_mask_debug = cv2.inRange(
            hsv,
            self.hsv_values['black']['low'],
            self.hsv_values['black']['high']
        )
        self.rectangle = self.detect_rectangle(self.black_mask_debug)

        # 红色激光 mask
        red_mask1 = cv2.inRange(
            hsv,
            self.hsv_values['red1']['low'],
            self.hsv_values['red1']['high']
        )
        red_mask2 = cv2.inRange(
            hsv,
            self.hsv_values['red2']['low'],
            self.hsv_values['red2']['high']
        )
        self.red_mask_debug = cv2.bitwise_or(red_mask1, red_mask2)
        self.red_point = self.detect_laser(self.red_mask_debug)

        # 绿色激光 mask
        self.green_mask_debug = cv2.inRange(
            hsv,
            self.hsv_values['green']['low'],
            self.hsv_values['green']['high']
        )
        self.green_point = self.detect_laser(self.green_mask_debug)

        # 统计时间和 FPS
        self.processing_time = time.time() - start_process
        self.frame_count += 1
        if time.time() - self.start_time > 1:
            self.fps = self.frame_count
            self.frame_count = 0
            self.start_time = time.time()

    def draw_info(self, frame):
        """
        重写父类的 draw_info 方法：
        - 先调用父类的绘制和信息逻辑
        - 不在此处 cv2.imshow 避免在子线程使用 GUI
        """
        super().draw_info(frame)

    def run(self):
        """
        重写父类的 run 方法：
        - 在主线程中进行摄像头读取与窗口显示
        - 子线程中调用 processing_loop 来处理图像
        - 主线程里额外多显示各调试用 mask
        """
        # 启动处理线程
        processing_thread = threading.Thread(target=self.processing_loop)
        processing_thread.start()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头帧，正在退出...")
                break

            # 将读取到的帧交给处理线程
            with self.lock:
                self.frame = frame

            # 拿到处理完的帧
            display_copy = None
            with self.lock:
                if self.display_frame is not None:
                    display_copy = self.display_frame.copy()

            # 在主线程中进行显示
            if display_copy is not None:
                cv2.imshow('Result', display_copy)

                # 在同一个主线程中显示调试用的 mask
                if self.black_mask_debug is not None:
                    cv2.imshow('Black Mask Debug', self.black_mask_debug)
                if self.red_mask_debug is not None:
                    cv2.imshow('Red Mask Debug', self.red_mask_debug)
                if self.green_mask_debug is not None:
                    cv2.imshow('Green Mask Debug', self.green_mask_debug)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 按 Q 退出
                self.running = False
                break
            elif key == ord('s'):  # 按 S 保存
                with self.lock:
                    self.save_hsv_to_json(self.hsv_config_path)
                    print("HSV 阈值已保存到 json 文件。")

        # 等待处理线程结束
        processing_thread.join()
        # 等待串口线程结束
        if self.serial_thread.is_alive():
            self.serial_thread.join()

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = DebugLaserTracker()
    tracker.run()