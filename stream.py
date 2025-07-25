from flask import Flask, render_template, Response
import json
import cv2
import numpy as np
import time
from datetime import datetime
import threading
import serial
import serial.tools.list_ports
from flask import jsonify

# Initialize Flask app
app = Flask(__name__)


class LaserTracker:
    def __init__(self):

        # setting.json 设置文件
        with open('setting.json', 'r') as json_file:
            data = json.load(json_file)
        url = data["url"]  # 摄像头配置
        hsv_config = data["hsv_config_path"]
        self.color = data["color"]

        # 配置串口
        serial_port = data["serial_port"]  # 串口设备名称
        baud_rate = data["baud_rate"]  # 波特率，根据你的设备设置
        timeout = data["timeout"]  # 超时时间（秒）

        self.scale_percent = data["scale_percent"]  # n%的缩放

        # 存储json地址
        self.hsv_config_path = hsv_config

        # 初始化HSV阈值（与原逻辑相同）
        self.hsv_values = self.load_hsv_values(self.hsv_config_path)

        # 初始化存储变量
        self.outer_rect = None
        self.inner_rect = None
        self.red_point = None
        self.green_point = None
        self.rectangle = None

        # 性能追踪变量
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        self.processing_time = 0

        # 创建调试窗口
        cv2.namedWindow('Result')
        self.create_trackbars()


        # 初始化摄像头
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # 多线程相关变量
        self.running = True             # 控制线程循环的开关
        self.frame = None              # 用于存放主线程读取的帧
        self.lock = threading.Lock()    # 用于线程间同步访问
        self.frame_lock = threading.Lock()


        # 处理过的帧用于在界面上显示（可能是带有信息绘制后的帧）
        self.display_frame = None

        # 打开串口
        try:
            self.ser = serial.Serial(serial_port, baud_rate, timeout=timeout)
            print("串口成功打开。")
        except Exception as e:
            self.ser = None
            print(f"串口打开失败： {e}")

        # 启动串口发送线程
        self.serial_thread = threading.Thread(target=self.serial_loop)
        self.serial_thread.start()

    def serial_loop(self):
        """
        新增线程:
        定期读取 outer_rect 的四个点以及红点和绿点的坐标 (降采样后乘以 2),
        以 "@[Message]/r/n" 形式发送到串口。
        """
        while self.running:
            with self.lock:
                # 发送外框四个点
                if self.outer_rect is not None:
                    # 取得外框四个点，乘以 2
                    corners = (self.outer_rect.reshape(-1, 2) * 2).astype(int)
                    # 拼接发送字符串
                    coords_str = ",".join([f"({pt[0]},{pt[1]})" for pt in corners])
                    message = f"@RECT:{coords_str}/r/n"
                    # 发送至串口
                    if self.ser and self.ser.is_open:
                        self.ser.write(message.encode('utf-8'))
                # 发送红色激光点
                if self.red_point is not None:
                    red_x, red_y = int(self.red_point[0] * 2), int(self.red_point[1] * 2)
                    red_message = f"@RED:({red_x},{red_y})/r/n"
                    if self.ser and self.ser.is_open:
                        self.ser.write(red_message.encode('utf-8'))
                # 发送绿色激光点
                if self.green_point is not None:
                    green_x, green_y = int(self.green_point[0] * 2), int(self.green_point[1] * 2)
                    green_message = f"@GREEN:({green_x},{green_y})/r/n"
                    if self.ser and self.ser.is_open:
                        self.ser.write(green_message.encode('utf-8'))
            # 每次循环间隔
            time.sleep(0.1)

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
        cv2.createTrackbar('Black H Low', 'Result',
                           int(self.hsv_values['black']['low'][0]), 255, nothing)
        cv2.createTrackbar('Black H High', 'Result',
                           int(self.hsv_values['black']['high'][0]), 255, nothing)
        cv2.createTrackbar('Black S Low', 'Result',
                           int(self.hsv_values['black']['low'][1]), 255, nothing)
        cv2.createTrackbar('Black S High', 'Result',
                           int(self.hsv_values['black']['high'][1]), 255, nothing)
        cv2.createTrackbar('Black V Low', 'Result',
                           int(self.hsv_values['black']['low'][2]), 255, nothing)
        cv2.createTrackbar('Black V High', 'Result',
                           int(self.hsv_values['black']['high'][2]), 255, nothing)

        cv2.createTrackbar('Red S Low', 'Result',
                           int(self.hsv_values['red1']['low'][1]), 255, nothing)
        cv2.createTrackbar('Red V Low', 'Result',
                           int(self.hsv_values['red1']['low'][2]), 255, nothing)

        cv2.createTrackbar('Green H Low', 'Result',
                           int(self.hsv_values['green']['low'][0]), 180, nothing)
        cv2.createTrackbar('Green H High', 'Result',
                           int(self.hsv_values['green']['high'][0]), 180, nothing)

    def update_hsv_values(self):
        """更新HSV阈值（从Trackbar获取）"""
        black_h_low = cv2.getTrackbarPos('Black H Low', 'Result')
        black_h_high = cv2.getTrackbarPos('Black H High', 'Result')
        black_s_low = cv2.getTrackbarPos('Black S Low', 'Result')
        black_s_high = cv2.getTrackbarPos('Black S High', 'Result')
        black_v_low = cv2.getTrackbarPos('Black V Low', 'Result')
        black_v_high = cv2.getTrackbarPos('Black V High', 'Result')

        red_s = cv2.getTrackbarPos('Red S Low', 'Result')
        red_v = cv2.getTrackbarPos('Red V Low', 'Result')
        green_h_low = cv2.getTrackbarPos('Green H Low', 'Result')
        green_h_high = cv2.getTrackbarPos('Green H High', 'Result')

        self.hsv_values['black']['low'][:] = [black_h_low, black_s_low, black_v_low]
        self.hsv_values['black']['high'][:] = [black_h_high, black_s_high, black_v_high]
        self.hsv_values['red1']['low'][1:] = [red_s, red_v]
        self.hsv_values['red2']['low'][1:] = [red_s, red_v]
        self.hsv_values['green']['low'][0] = green_h_low
        self.hsv_values['green']['high'][0] = green_h_high

    def processing_loop(self):
        """
        独立线程：
        - 获取最新帧
        - 执行图像处理
        - 更新追踪信息
        - 将结果帧保存到 display_frame
        """
        while self.running:
            # 检查是否有新帧可用
            frame_to_process = None
            with self.lock:
                if self.frame is not None:
                    # 复制一份，避免直接操作 self.frame
                    frame_to_process = self.frame.copy()

            if frame_to_process is not None:
                # 处理帧
                self.process_frame(frame_to_process)
                # 绘制信息
                self.draw_info(frame_to_process)


                width = int(frame_to_process.shape[1] * self.scale_percent / 100)
                height = int(frame_to_process.shape[0] * self.scale_percent / 100)
                resized_frame = cv2.resize(frame_to_process, (width, height))

                # 将处理完的帧存放以便主线程展示
                with self.lock:
                    self.display_frame = resized_frame
            else:
                # 若没有可用帧，稍微休眠
                time.sleep(0.01)

    def process_frame(self, frame):
        """单帧处理，与原先逻辑相同"""
        start_process = time.time()
        frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)

        self.update_hsv_values()

        black_mask = cv2.inRange(hsv, self.hsv_values['black']['low'], self.hsv_values['black']['high'])
        self.rectangle = self.detect_rectangle(black_mask)

        red_mask1 = cv2.inRange(hsv, self.hsv_values['red1']['low'], self.hsv_values['red1']['high'])
        red_mask2 = cv2.inRange(hsv, self.hsv_values['red2']['low'], self.hsv_values['red2']['high'])
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        self.red_point = self.detect_laser(red_mask)

        green_mask = cv2.inRange(hsv, self.hsv_values['green']['low'], self.hsv_values['green']['high'])
        self.green_point = self.detect_laser(green_mask)

        # 计算处理时间和FPS
        self.processing_time = time.time() - start_process
        self.frame_count += 1
        if time.time() - self.start_time > 1:
            self.fps = self.frame_count
            self.frame_count = 0
            self.start_time = time.time()

    def draw_info(self, frame):
        """在画面上绘制信息，与原先逻辑相同"""
        info_list = [
            f"DateTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"FPS: {self.fps}",
            f"Process Time: {self.processing_time:.3f}s",
            f"Frame Size: {frame.shape[1]}x{frame.shape[0]}"
        ]

        if self.outer_rect is not None:
            # 绘制外框
            cv2.drawContours(frame, [self.outer_rect * 2], 0, (0, 255, 0), 2)
            if self.inner_rect is not None:
                cv2.drawContours(frame, [self.inner_rect * 2], 0, (0, 0, 255), 2)
                outer_corners = self.outer_rect.reshape(-1, 2)
                inner_corners = self.inner_rect.reshape(-1, 2)
                info_list.append(f"Outer Rectangle: {outer_corners}")
                info_list.append(f"Inner Rectangle: {inner_corners}")
            else:
                corners = self.outer_rect.reshape(-1, 2)
                info_list.append(f"Rectangle Corners: {corners}")

        # 红色激光点
        if self.red_point is not None:
            red_x, red_y = self.red_point[0] * 2, self.red_point[1] * 2
            cv2.rectangle(frame, (red_x - 10, red_y - 10),
                          (red_x + 10, red_y + 10), (0, 0, 255), -1)
            position = self.check_point_position(
                (red_x, red_y),
                self.rectangle * 2 if self.rectangle is not None else None
            )
            info_list.append(f"Red Laser: ({red_x}, {red_y}) - {position}")

        # 绿色激光点
        if self.green_point is not None:
            green_x, green_y = self.green_point[0] * 2, self.green_point[1] * 2
            cv2.rectangle(frame, (green_x - 10, green_y - 10),
                          (green_x + 10, green_y + 10), (0, 255, 0), -1)
            info_list.append(f"Green Laser: ({green_x}, {green_y})")

        # 如果红绿都存在，计算激光点之间距离
        if self.red_point is not None and self.green_point is not None:
            dx = (self.green_point[0] - self.red_point[0]) * 2
            dy = (self.green_point[1] - self.red_point[1]) * 2
            distance = np.sqrt(dx * dx + dy * dy)
            info_list.append(f"Laser Distance: {distance:.2f}")
            cv2.line(frame,
                     (red_x, red_y), (green_x, green_y),
                     (255, 255, 0), 2)
            if distance < 30:
                text_x = 10
                text_y = frame.shape[0] - 40
                text = "OVERLAP!"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.rectangle(
                    frame,
                    (text_x, text_y - text_height - 5),
                    (text_x + text_width + 10, text_y + 5),
                    (0, 0, 0),
                    -1
                )
                cv2.putText(
                    frame,
                    text,
                    (text_x + 5, text_y),
                    font,
                    font_scale,
                    (0, 255, 255),
                    thickness
                )

        # 显示文本信息
        for i, info in enumerate(info_list):
            cv2.putText(frame, info, (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    def detect_rectangle(self, mask):
        """检测黑色区域内的外框和内框，与原先逻辑相同"""
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        inner_rect = None
        outer_rect = None
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for contour in contours[:2]:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:
                    if outer_rect is None:
                        outer_rect = approx
                    else:
                        inner_rect = approx
                        break
        self.inner_rect = inner_rect
        self.outer_rect = outer_rect
        return outer_rect

    def detect_laser(self, mask):
        """检测激光点，对应红色或绿色"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_contour = max(contours, key=cv2.contourArea)

            # 求相关质心
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)

        return None

    def check_point_position(self, point, rectangle):
        """判断激光点相对于外框/内框的位置"""
        if point is None or self.outer_rect is None:
            return "unknown"
        outer_rect = self.outer_rect * 2
        inner_rect = self.inner_rect * 2 if self.inner_rect is not None else None
        outer_result = cv2.pointPolygonTest(outer_rect, point, False)

        if inner_rect is not None:
            inner_result = cv2.pointPolygonTest(inner_rect, point, False)
            if outer_result >= 0:
                if inner_result > 0:
                    return "inside"
                elif inner_result == 0:
                    return "between"
                else:
                    return "between"
            else:
                return "outside"
        else:
            # 若无内框，仅判断外框
            if outer_result > 0:
                return "inside"
            elif outer_result < 0:
                return "outside"
            else:
                return "between"

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
        """
        主线程负责：
        - 从摄像头读取帧
        - 显示最终处理结果 (display_frame)
        - 处理用户输入（退出程序等）
        """

        # 启动处理线程
        processing_thread = threading.Thread(target=self.processing_loop)
        processing_thread.start()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头帧，正在退出...")
                break

            

            # 将最新的帧交给处理线程
            with self.lock:
                self.frame = frame

            # 如果处理线程已经对帧进行了处理，则展示
            display_copy = None
            with self.lock:
                if self.display_frame is not None:
                    display_copy = self.display_frame.copy()
            # 需要考虑共享访问的问题，所以说先给 self.display_frame 复制出来再显示，否则可能会出问题


            if display_copy is not None:

                cv2.imshow('Result', display_copy)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 按 Q 退出程序
                self.running = False
                break
            elif key == ord('s'):  # 按 S 保存当前 HSV 值到 JSON 文件
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

    def generate_frames(self):
        """Generator function for streaming frames"""
        while self.running:
            # Wait for processed frame
            display_copy = None
            with self.lock:
                if self.display_frame is not None:
                    display_copy = self.display_frame.copy()

            if display_copy is not None:
                # Encode the frame for web streaming
                _, buffer = cv2.imencode('.jpg', display_copy,[cv2.IMWRITE_JPEG_QUALITY, 50])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.01)  # Small delay to prevent excessive CPU usage

    def run_with_web(self):
        """Run the tracker with web streaming instead of local display"""
        # Start processing thread
        processing_thread = threading.Thread(target=self.processing_loop)
        processing_thread.start()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Cannot read camera frame, exiting...")
                break

            with self.lock:
                if (self.color == 1):
                    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                # 仅在Linux上可能出现的错误使用
                self.frame = frame

            time.sleep(0.01)

        # Cleanup
        processing_thread.join()
        if self.serial_thread.is_alive():
            self.serial_thread.join()
        self.cap.release()

    def get_info_dict(self):
        """Return current tracking information as a dictionary"""
        info_dict = {
            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'fps': int(self.fps),  # 确保是普通的Python整数
            'process_time': f'{self.processing_time:.3f}s',
            'frame_size': f'{self.cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}'
        }

        if self.outer_rect is not None:
            # 确保转换为普通的Python列表
            outer_corners = (self.outer_rect.reshape(-1, 2) * 2).tolist()
            info_dict['outer_rectangle'] = outer_corners
        else:
            # 如果 outer_rect 为 None，则清空 outer_rectangle
            info_dict['outer_rectangle'] = []

        if self.inner_rect is not None:
            # 确保转换为普通的Python列表
            inner_corners = (self.inner_rect.reshape(-1, 2) * 2).tolist()
            info_dict['inner_rectangle'] = inner_corners
        else:
            # 如果 inner_rect 为 None，则清空 inner_rectangle
            info_dict['inner_rectangle'] = []

        if self.red_point is not None:
            red_x, red_y = int(self.red_point[0] * 2), int(self.red_point[1] * 2)  # 转换为普通的Python整数
            position = self.check_point_position(
                (red_x, red_y),
                self.rectangle * 2 if self.rectangle is not None else None
            )
            info_dict['red_laser'] = {
                'x': red_x,
                'y': red_y,
                'position': str(position)  # 确保position是字符串
            }
        else:
            info_dict['red_laser'] = {
                'x': None,
                'y': None,
                'position': ' Unknown'
            }

        # 绿点处理
        if self.green_point is not None:
            green_x, green_y = int(self.green_point[0] * 2), int(self.green_point[1] * 2)  # 转换为普通的Python整数
            info_dict['green_laser'] = {
                'x': green_x,
                'y': green_y
            }
        else:
            info_dict['green_laser'] = {
                'x': None,
                'y': None
            }

        # 只有当两个点都存在时才计算距离和重叠状态
        if self.red_point is not None and self.green_point is not None:
            dx = float(green_x - red_x)  # 确保是Python float
            dy = float(green_y - red_y)  # 确保是Python float
            distance = float(np.sqrt(dx * dx + dy * dy))  # 确保是Python float
            info_dict['laser_distance'] = f'{distance:.2f}'
            info_dict['lasers_overlap'] = bool(distance < 30)  # 转换为Python bool
        else:
            info_dict['laser_distance'] = None
            info_dict['lasers_overlap'] = False

        return info_dict


# Flask routes
@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(tracker.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_info')
def get_info():
    """Return current tracking information as JSON"""
    return jsonify(tracker.get_info_dict())


if __name__ == "__main__":
    tracker = LaserTracker()

    # Start the tracker in a separate thread
    tracker_thread = threading.Thread(target=tracker.run_with_web)
    tracker_thread.daemon = True
    tracker_thread.start()

    # Run the Flask app
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)