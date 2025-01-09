import cv2
import numpy as np
import time
from datetime import datetime
import threading

class LaserTracker:
    def __init__(self):
        # 初始化 HSV 阈值
        self.hsv_values = {
            'black': {
                'low': np.array([0, 0, 0]),    # 黑色区域
                'high': np.array([180, 70, 60]) # 黑色上限
            }
        }

        # 性能追踪变量
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        self.processing_time = 0

        # 窗口名称
        self.result_window = 'Result'
        self.mask_window = 'Black Mask'
        cv2.namedWindow(self.result_window)

        # 初始化摄像头
        self.cap = cv2.VideoCapture(1)  # 根据需要更改摄像头索引
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 线程/共享资源控制
        self.running = True
        self.lock = threading.Lock()
        self.latest_frame = None       # 从捕获线程获取的原始帧
        self.display_frame = None      # 处理完成后要显示的结果
        self.mask_for_display = None   # 供显示的掩码
        self.rectangle = None          # 检测到的矩形

        # 创建 Trackbars
        self.create_trackbars()
        # 存储从主线程读取到的 HSV 阈值，以供处理线程访问
        self.current_hsv_low = self.hsv_values['black']['low'].copy()
        self.current_hsv_high = self.hsv_values['black']['high'].copy()

    def create_trackbars(self):
        """在窗口中创建 Trackbar 用于动态调节 HSV 阈值"""
        def nothing(x):
            pass

        cv2.createTrackbar('Black H Low', self.result_window,
                           int(self.hsv_values['black']['low'][0]), 255, nothing)
        cv2.createTrackbar('Black H High', self.result_window,
                           int(self.hsv_values['black']['high'][0]), 255, nothing)
        cv2.createTrackbar('Black S Low', self.result_window,
                           int(self.hsv_values['black']['low'][1]), 255, nothing)
        cv2.createTrackbar('Black S High', self.result_window,
                           int(self.hsv_values['black']['high'][1]), 255, nothing)
        cv2.createTrackbar('Black V Low', self.result_window,
                           int(self.hsv_values['black']['low'][2]), 255, nothing)
        cv2.createTrackbar('Black V High', self.result_window,
                           int(self.hsv_values['black']['high'][2]), 255, nothing)

    def read_trackbars_in_main_thread(self):
        """
        在主线程中读取当前 Trackbar 值，并保存到共享变量 self.current_hsv_low/high。
        这能更快地更新滑块值，减少延迟。
        """
        black_h_low = cv2.getTrackbarPos('Black H Low', self.result_window)
        black_h_high = cv2.getTrackbarPos('Black H High', self.result_window)
        black_s_low = cv2.getTrackbarPos('Black S Low', self.result_window)
        black_s_high = cv2.getTrackbarPos('Black S High', self.result_window)
        black_v_low = cv2.getTrackbarPos('Black V Low', self.result_window)
        black_v_high = cv2.getTrackbarPos('Black V High', self.result_window)

        # 用锁来同步更新当前的低/高阈值
        with self.lock:
            self.current_hsv_low = np.array([black_h_low, black_s_low, black_v_low])
            self.current_hsv_high = np.array([black_h_high, black_s_high, black_v_high])

    def detect_rectangle(self, mask):
        """检测最大面积轮廓并判断是否为矩形"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(max_contour, True)
            approx = cv2.approxPolyDP(max_contour, epsilon, True)
            if len(approx) == 4:
                return approx
        return None

    def process_frame(self, frame):
        """
        对帧进行高斯模糊、HSV 转换以及生成黑色掩码，
        同时检测矩形，更新处理时间
        """
        start_process = time.time()

        # 高斯模糊
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 从共享变量里获取最新阈值
        with self.lock:
            hsv_low = self.current_hsv_low.copy()
            hsv_high = self.current_hsv_high.copy()

        # 生成黑色掩码
        black_mask = cv2.inRange(hsv, hsv_low, hsv_high)

        # 检测矩形
        rect = self.detect_rectangle(black_mask)
        self.rectangle = rect

        self.processing_time = time.time() - start_process
        return black_mask

    def draw_info(self, frame, black_mask):
        """
        在 frame 上绘制信息（FPS、处理耗时、HSV 参数等），
        并返回最终可显示的图像
        """
        # 绘制文本信息
        # 从共享变量中获取最新 HSV 阈值仅用于显示
        with self.lock:
            hsv_low_display = self.current_hsv_low.copy()
            hsv_high_display = self.current_hsv_high.copy()

        info_list = [
            f"FPS: {self.fps}",
            f"Process Time: {self.processing_time:.3f}s",
            f"Frame Size: {frame.shape[1]}x{frame.shape[0]}",
            f"Black HSV Low: {hsv_low_display}",
            f"Black HSV High: {hsv_high_display}"
        ]

        # 如果检测到矩形，则绘制
        if self.rectangle is not None:
            corners = self.rectangle.reshape(-1, 2)
            info_list.append(f"Rectangle Corners: {corners}")
            cv2.drawContours(frame, [self.rectangle], 0, (0, 255, 0), 2)

        # 将信息写到帧上
        for i, info in enumerate(info_list):
            cv2.putText(
                frame, info, (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2
            )

        return frame

    def capture_thread(self):
        """
        线程1：捕获摄像头图像，每次读取后放入 shared 变量 latest_frame
        """
        while self.running:
            ret, frame = self.cap.read()

            # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 仅在Linux上可能出现的错误使用

            if not ret:
                print("无法读取摄像头数据，结束捕获线程...")
                self.running = False
                break

            with self.lock:
                self.latest_frame = frame.copy()

            time.sleep(0.001)  # 防止忙等导致CPU过高

    def processing_thread(self):
        """
        线程2：等待最新帧 -> 处理图像 -> 生成可显示结果和掩码
        """
        local_frame_count = 0
        local_start_time = time.time()

        while self.running:
            frame_to_process = None
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()

            if frame_to_process is not None:
                # 执行图像处理并绘制信息
                black_mask = self.process_frame(frame_to_process)
                result_frame = self.draw_info(frame_to_process, black_mask)

                # 计算 FPS
                local_frame_count += 1
                if time.time() - local_start_time >= 1.0:
                    self.fps = local_frame_count
                    local_frame_count = 0
                    local_start_time = time.time()

                # 将处理后的帧和掩码存入共享变量
                with self.lock:
                    self.display_frame = result_frame.copy()
                    self.mask_for_display = black_mask.copy()
            else:
                time.sleep(0.001)

    def run(self):
        """
        主线程：负责显示结果帧 (display_frame) 和黑色掩码 (mask_for_display)，
        主线程中也读取 Trackbar 值以减少滑块延迟，并监听键盘事件来退出程序
        """
        # 启动捕获线程和处理线程
        t1 = threading.Thread(target=self.capture_thread)
        t2 = threading.Thread(target=self.processing_thread)
        t1.start()
        t2.start()

        while True:
            # 每次循环都读取滑块，及时更新阈值
            self.read_trackbars_in_main_thread()

            frame_for_display = None
            mask_for_display = None

            # 获取处理线程提供的帧/掩码
            with self.lock:
                if self.display_frame is not None:
                    frame_for_display = self.display_frame.copy()
                if self.mask_for_display is not None:
                    mask_for_display = self.mask_for_display.copy()

            # 在主线程中进行显示
            if frame_for_display is not None:
                cv2.imshow(self.result_window, frame_for_display)
            if mask_for_display is not None:
                cv2.imshow(self.mask_window, mask_for_display)

            # 检测退出事件
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        # 等待其他线程结束，并释放资源
        t1.join()
        t2.join()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = LaserTracker()
    tracker.run()