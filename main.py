import json
import cv2
import numpy as np
# import serial
import time
from datetime import datetime


class LaserTracker:
    def __init__(self):

        # 存储json地址
        self.hsv_config_path = 'hsv_values.json'

        # 初始化HSV阈值（与原逻辑相同）
        self.hsv_values = self.load_hsv_values(self.hsv_config_path)

        # 初始化存储变量
        self.outer_rect = None
        self.inner_rect = None
        self.red_point = None
        self.green_point = None

        # 性能追踪变量
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        self.processing_time = 0

        # 创建调试窗口
        cv2.namedWindow('Result')
        self.create_trackbars()

        # # 初始化串口
        # try:
        #     self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
        # except:
        #     print("串口连接失败")
        #     self.ser = None

        # 初始化摄像头
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 降低分辨率提高性能
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 初始化存储变量
        self.rectangle = None
        self.red_point = None
        self.green_point = None

    def load_hsv_values(self, file_path):
        """从文件加载 HSV 阈值"""
        try:
            with open(file_path, 'r') as file:
                hsv_data = json.load(file)

            # 将列表转换为 NumPy 数组
            for key, value in hsv_data.items():
                hsv_data[key]['low'] = np.array(hsv_data[key]['low'])
                hsv_data[key]['high'] = np.array(hsv_data[key]['high'])

            return hsv_data
        except Exception as e:
            print(f"Error loading HSV values: {e}")
            return None

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

        # 红色阈值调节
        cv2.createTrackbar('Red S Low', 'Result', int(self.hsv_values['red1']['low'][1]), 255, nothing)
        cv2.createTrackbar('Red V Low', 'Result', int(self.hsv_values['red1']['low'][2]), 255, nothing)

        # 绿色阈值调节
        cv2.createTrackbar('Green H Low', 'Result', int(self.hsv_values['green']['low'][0]), 180, nothing)
        cv2.createTrackbar('Green H High', 'Result', int(self.hsv_values['green']['high'][0]), 180, nothing)

    def update_hsv_values(self):
        """更新HSV阈值"""
        # 使用trackbar的值更新HSV阈值
        black_h_low = cv2.getTrackbarPos('Black H Low', 'Result')
        black_h_high = cv2.getTrackbarPos('Black H High', 'Result')
        black_v_low = cv2.getTrackbarPos('Black V Low', 'Result')
        black_v_high = cv2.getTrackbarPos('Black V High', 'Result')
        black_s_low = cv2.getTrackbarPos('Black S Low', 'Result')
        black_s_high = cv2.getTrackbarPos('Black S High', 'Result')

        red_s = cv2.getTrackbarPos('Red S Low', 'Result')
        red_v = cv2.getTrackbarPos('Red V Low', 'Result')
        green_h_low = cv2.getTrackbarPos('Green H Low', 'Result')
        green_h_high = cv2.getTrackbarPos('Green H High', 'Result')

        self.hsv_values['black']['low'][:] = [black_h_low, black_s_low, black_v_low]
        self.hsv_values['black']['high'][:] = [black_h_high,black_s_high, black_v_high]
        self.hsv_values['red1']['low'][1:] = [red_s, red_v]
        self.hsv_values['red2']['low'][1:] = [red_s, red_v]
        self.hsv_values['green']['low'][0] = green_h_low
        self.hsv_values['green']['high'][0] = green_h_high

    def process_frame(self, frame):
        """处理单帧图像"""
        # 计时开始
        start_process = time.time()

        # 图像预处理
        frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # 降采样
        hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)

        # 更新HSV值
        self.update_hsv_values()

        # 检测黑色矩形
        black_mask = cv2.inRange(hsv, self.hsv_values['black']['low'], self.hsv_values['black']['high'])
        self.rectangle = self.detect_rectangle(black_mask)

        # 检测激光点
        red_mask1 = cv2.inRange(hsv, self.hsv_values['red1']['low'], self.hsv_values['red1']['high'])
        red_mask2 = cv2.inRange(hsv, self.hsv_values['red2']['low'], self.hsv_values['red2']['high'])
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        self.red_point = self.detect_laser(red_mask)

        green_mask = cv2.inRange(hsv, self.hsv_values['green']['low'], self.hsv_values['green']['high'])
        self.green_point = self.detect_laser(green_mask)

        # 计算处理时间
        self.processing_time = time.time() - start_process

        # 更新FPS
        self.frame_count += 1
        if time.time() - self.start_time > 1:
            self.fps = self.frame_count
            self.frame_count = 0
            self.start_time = time.time()

    def draw_info(self, frame):
        """在画面上绘制信息"""
        # 基础信息
        info_list = [
            f"DateTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"FPS: {self.fps}",
            f"Process Time: {self.processing_time:.3f}s",
            f"Frame Size: {frame.shape[1]}x{frame.shape[0]}"
        ]

        # 绘制矩形信息
        if self.outer_rect is not None:
            # 绘制外框
            cv2.drawContours(frame, [self.outer_rect * 2], 0, (0, 255, 0), 2)

            # 绘制内框
            if self.inner_rect is not None:
                cv2.drawContours(frame, [self.inner_rect * 2], 0, (0, 0, 255), 2)

            # 添加矩形信息到info_list
            if self.inner_rect is not None:
                outer_corners = self.outer_rect.reshape(-1, 2)
                inner_corners = self.inner_rect.reshape(-1, 2)
                info_list.append(f"Outer Rectangle: {outer_corners}")
                info_list.append(f"Inner Rectangle: {inner_corners}")
            else:
                corners = self.outer_rect.reshape(-1, 2)
                info_list.append(f"Rectangle Corners: {corners}")

        # 激光点信息
        if self.red_point is not None:
            red_x, red_y = self.red_point[0] * 2, self.red_point[1] * 2  # *2因为之前做了降采样
            cv2.rectangle(frame, (red_x - 10, red_y - 10), (red_x + 10, red_y + 10), (0, 0, 255), -1)
            position = self.check_point_position((red_x, red_y),
                                                 self.rectangle * 2 if self.rectangle is not None else None)
            info_list.append(f"Red Laser: ({red_x}, {red_y}) - {position}")

        if self.green_point is not None:
            green_x, green_y = self.green_point[0] * 2, self.green_point[1] * 2
            cv2.rectangle(frame, (green_x - 10, green_y - 10), (green_x + 10, green_y + 10), (0, 255, 0), -1)
            info_list.append(f"Green Laser: ({green_x}, {green_y})")

        if self.red_point is not None:
            if self.green_point is not None:
                # 计算和显示激光点之间的距离
                dx = green_x - red_x
                dy = green_y - red_y
                distance = np.sqrt(dx * dx + dy * dy)
                info_list.append(f"Laser Distance: {distance:.2f}")

                # 添加距离线
                cv2.line(frame, (red_x, red_y), (green_x, green_y), (255, 255, 0), 2)

                # 改进OVERLAP显示
                if distance < 30:
                    # 确定文本位置
                    text_x = 10
                    text_y = frame.shape[0] - 40

                    # 添加黑色背景提高可读性
                    text = "OVERLAP!"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    thickness = 2
                    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

                    cv2.rectangle(frame,
                                  (text_x, text_y - text_height - 5),
                                  (text_x + text_width + 10, text_y + 5),
                                  (0, 0, 0),
                                  -1)

                    # 绘制OVERLAP文本
                    cv2.putText(frame,
                                text,
                                (text_x + 5, text_y),
                                font,
                                font_scale,
                                (0, 255, 255),
                                thickness)

        # 绘制信息列表
        for i, info in enumerate(info_list):
            cv2.putText(frame, info, (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    def detect_rectangle(self, mask):
        """检测内外矩形"""
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        inner_rect = None
        outer_rect = None

        if contours:
            # 按面积排序轮廓
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for contour in contours[:2]:  # 只处理最大的两个轮廓
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 4:  # 确保是四边形
                    if outer_rect is None:
                        outer_rect = approx
                    else:
                        inner_rect = approx
                        break

        self.inner_rect = inner_rect  # 保存内框
        self.outer_rect = outer_rect  # 保存外框
        return outer_rect  # 为保持兼容性，返回外框

    def detect_laser(self, mask):
        """检测激光点"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(max_contour)

            # 求相关质心，不用霍夫圆变换的原因是，激光点极易HSV识别下成为不规则点块，霍夫圆变换极易识别失败
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)

        return None

    def check_point_position(self, point, rectangle):
        """检查点相对于内外矩形的位置"""
        if point is None or self.outer_rect is None:
            return "unknown"

        # 将坐标转换为相同比例
        outer_rect = self.outer_rect * 2
        inner_rect = self.inner_rect * 2 if self.inner_rect is not None else None

        # 检查外框
        outer_result = cv2.pointPolygonTest(outer_rect, point, False)

        # 如果有内框，检查内框
        if inner_rect is not None:
            inner_result = cv2.pointPolygonTest(inner_rect, point, False)

            if outer_result >= 0:  # 点在外框内或边上
                if inner_result > 0:  # 点在内框内
                    return "inside"
                elif inner_result == 0:  # 点在内框边上
                    return "between"
                else:  # 点在胶带区域
                    return "between"
            else:  # 点在外框外
                return "outside"
        else:
            # 如果没有检测到内框，仅判断外框
            return "inside" if outer_result > 0 else "outside" if outer_result < 0 else "between"

    def save_hsv_to_json(self, file_path):
        """将当前 HSV 阈值保存到 JSON 文件"""
        # 将 numpy.ndarray 转换为 list
        hsv_values_serializable = {
            key: {
                sub_key: value.tolist() for sub_key, value in sub_dict.items()
            } for key, sub_dict in self.hsv_values.items()
        }

        # 保存到 JSON 文件
        with open(file_path, 'w') as json_file:
            json.dump(hsv_values_serializable, json_file, indent=4)

    def run(self):
        """主循环"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 仅在Linux上可能出现的错误使用

            self.process_frame(frame)
            self.draw_info(frame)

            cv2.imshow('Result', frame)

            # 监听键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 按 Q 退出程序
                break
            elif key == ord('s'):  # 按 S 保存当前 HSV 值到 JSON 文件
                self.save_hsv_to_json(self.hsv_config_path)
                print("HSV 阈值已保存到 json 文件。")

        self.cap.release()
        cv2.destroyAllWindows()
        # if self.ser:
        #     self.ser.close()



if __name__ == "__main__":
    tracker = LaserTracker()
    tracker.run()