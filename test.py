import cv2

def list_camera_indexes():
    """
    列出所有可用摄像头的索引。
    """
    available_indexes = []
    for index in range(10):  # 假设最多有 10 个摄像头
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_indexes.append(index)
            cap.release()
    return available_indexes


if __name__ == "__main__":
    # 获取所有摄像头的索引
    camera_indexes = list_camera_indexes()

    if not camera_indexes:
        print("未检测到任何摄像头。")
    else:
        print(f"检测到以下摄像头索引：{camera_indexes}")

        # 输出摄像头的 URL
        for index in camera_indexes:
            print(f"摄像头索引 {index} 的 URL 是：{index}")