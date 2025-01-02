import cv2
import numpy as np
import time
from datetime import datetime


class LaserTracker:
    def __init__(self):
        # 初始化HSV阈值
        self.hsv_values = {
            'black': {
                'low': np.array([0, 0, 0]),  # 黑色区域
                'high': np.array([180, 70, 60])  # 黑色上限
            }
        }

        # 性能追踪变量
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        self.processing_time = 0

        # 创建窗口
        cv2.namedWindow('Result')
        cv2.namedWindow('Black Mask')
        self.create_trackbars()

        # 初始化摄像头
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 初始化存储变量
        self.rectangle = None
        self.red_point = None
        self.green_point = None

    def create_trackbars(self):
        """创建HSV调节滑块"""

        def nothing(x):
            pass

        # 黑色阈值调节 - 使用hsv_values中的初始值
        cv2.createTrackbar('Black H Low', 'Result', int(self.hsv_values['black']['low'][0]), 255, nothing)
        cv2.createTrackbar('Black H High', 'Result', int(self.hsv_values['black']['high'][0]), 255, nothing)
        cv2.createTrackbar('Black S Low', 'Result', int(self.hsv_values['black']['low'][1]), 255, nothing)
        cv2.createTrackbar('Black S High', 'Result', int(self.hsv_values['black']['high'][1]), 255, nothing)
        cv2.createTrackbar('Black V Low', 'Result', int(self.hsv_values['black']['low'][2]), 255, nothing)
        cv2.createTrackbar('Black V High', 'Result', int(self.hsv_values['black']['high'][2]), 255, nothing)

    def update_hsv_values(self):
        """更新HSV阈值"""
        # 黑色阈值
        black_h_low = cv2.getTrackbarPos('Black H Low', 'Result')
        black_h_high = cv2.getTrackbarPos('Black H High', 'Result')
        black_s_low = cv2.getTrackbarPos('Black S Low', 'Result')
        black_s_high = cv2.getTrackbarPos('Black S High', 'Result')
        black_v_low = cv2.getTrackbarPos('Black V Low', 'Result')
        black_v_high = cv2.getTrackbarPos('Black V High', 'Result')

        self.hsv_values['black']['low'] = np.array([black_h_low, black_s_low, black_v_low])
        self.hsv_values['black']['high'] = np.array([black_h_high, black_s_high, black_v_high])

    def process_frame(self, frame):
        """处理单帧图像"""
        start_process = time.time()


        # 添加图像预处理步骤
        # 1. 高斯模糊去噪
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # 2. 转换到HSV空间
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 更新HSV值（每3帧更新一次）
        if self.frame_count % 3 == 0:
            self.update_hsv_values()

        # 生成黑色掩码
        black_mask = cv2.inRange(hsv, self.hsv_values['black']['low'], self.hsv_values['black']['high'])

        # 显示黑色mask
        cv2.imshow('Black Mask', black_mask)

        # 检测矩形
        self.rectangle = self.detect_rectangle(black_mask)

        self.processing_time = time.time() - start_process

    def detect_rectangle(self, mask):
        """检测矩形"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(max_contour, True)
            approx = cv2.approxPolyDP(max_contour, epsilon, True)
            if len(approx) == 4:
                return approx
        return None

    def draw_info(self, frame):
        """在画面上绘制信息"""
        # 基础信息
        info_list = [
            f"FPS: {self.fps}",
            f"Process Time: {self.processing_time:.3f}s",
            f"Frame Size: {frame.shape[1]}x{frame.shape[0]}",
            f"Black HSV Low: {self.hsv_values['black']['low']}",
            f"Black HSV High: {self.hsv_values['black']['high']}"
        ]

        # 矩形信息
        if self.rectangle is not None:
            corners = self.rectangle.reshape(-1, 2)
            info_list.append(f"Rectangle Corners: {corners}")
            cv2.drawContours(frame, [self.rectangle], 0, (0, 255, 0), 2)

        # 绘制信息
        for i, info in enumerate(info_list):
            cv2.putText(frame, info, (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    def run(self):
        """主循环"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.process_frame(frame)
            self.draw_info(frame)

            # 更新FPS
            self.frame_count += 1
            if time.time() - self.start_time > 1:
                self.fps = self.frame_count
                self.frame_count = 0
                self.start_time = time.time()

            cv2.imshow('Result', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = LaserTracker()
    tracker.run()