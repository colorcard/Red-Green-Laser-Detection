import serial
import serial.tools.list_ports
import json

# 获取所有串口设备实例。
# 如果没找到串口设备，则输出：“无串口设备。”
# 如果找到串口设备，则依次输出每个设备对应的串口号和描述信息。
ports_list = list(serial.tools.list_ports.comports())
if len(ports_list) <= 0:
    print("无串口设备。")
else:
    print("可用的串口设备如下：")
    for comport in ports_list:
        print(list(comport)[0], list(comport)[1])


# setting.json 设置文件
    with open('setting.json', 'r') as json_file:
        data = json.load(json_file)

    # 配置串口
serial_port = data["serial_port"]  # 串口设备名称
baud_rate = data["baud_rate"]  # 波特率，根据你的设备设置
timeout = data["timeout"]  # 超时时间（秒）

# 打开串口
try:
    ser = serial.Serial(port=serial_port, baudrate=baud_rate, timeout=timeout)
    print(f"成功打开串口 {serial_port}")

    # 测试写入数据
    ser.write(b"Hello, Serial Port!\n")
    print("数据已发送到串口")

    # 测试读取数据
    while True:
        if ser.in_waiting:  # 检查是否有数据
            data = ser.readline().decode('utf-8').strip()
            print(f"接收到的数据: {data}")

except serial.SerialException as e:
    print(f"打开串口失败: {e}")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("串口已关闭")