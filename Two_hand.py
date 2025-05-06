# Dự án điều khiển ô tô mô hình bằng cử chỉ tay
# Đầu vào: camera máy tính
# Đầu ra: điều khiển ô tô mô hình
# Đây là file test để kiểm tra các cử chỉ tay có thể nhận diện được hay không
# Các cử chỉ tay được nhận diện: số ngón tay
# Thư viện: OpenCV, MediaPipe, NumPy

import cv2
import mediapipe as mp
import numpy as np
import math
import time
from PIL import ImageFont, ImageDraw, Image
from collections import deque, Counter

# Font Unicode hỗ trợ tiếng Việt
FONT_PATH = "arial.ttf"  # Thay bằng font có hỗ trợ Unicode nếu cần

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# giảm độ nhạy của cử chỉ tay
MODE_BUFFER_SIZE = 20  # Tăng số khung hình để ổn định hơn khi chuyển chế độ
mode_buffer = deque([0]*MODE_BUFFER_SIZE, maxlen=MODE_BUFFER_SIZE)
FINGER_BUFFER_SIZE = 30  # Tăng số khung hình để ổn định hơn thay đổi cử chỉ tay
finger_buffer = deque([0]*FINGER_BUFFER_SIZE, maxlen=FINGER_BUFFER_SIZE)

def detect_hand_mode(num_hands):
    """
    Trả về chế độ:
        0 - Không có bàn tay
        1 - Có 1 bàn tay (điều khiển tốc độ)
        2 - Có 2 bàn tay (điều khiển góc lái)
    """
    if num_hands == 0:
        return 0
    elif num_hands == 1:
        return 1
    elif num_hands == 2:
        return 2
    else:
        return 0  # Trường hợp khác không xác định

def count_fingers(hand_landmarks):
    """
    Đếm số ngón tay đang duỗi, KHÔNG tính ngón cái.
    Trả về: số lượng ngón tay mở (từ 0 đến 4).
    """
    lm = hand_landmarks.landmark
    fingers = 0

    # Danh sách các ngón: trỏ, giữa, áp út, út (tip - 2 = pip)
    tip_ids = [8, 12, 16, 20]
    for tip_id in tip_ids:
        if lm[tip_id].y < lm[tip_id - 2].y:
            fingers += 1

    return fingers


def get_steering_angle_wheel_style(hand_landmarks_1, hand_landmarks_2, image_width, image_height):
    """
    Xác định góc lái dựa trên vị trí cổ tay của 2 tay như đang cầm vô lăng.
    Trả về: (angle_left, angle_right)
    """
    # Lấy cổ tay của 2 tay
    lm1 = hand_landmarks_1.landmark
    lm2 = hand_landmarks_2.landmark

    x1, y1 = lm1[0].x * image_width, lm1[0].y * image_height
    x2, y2 = lm2[0].x * image_width, lm2[0].y * image_height

    # Tính chênh lệch chiều dọc giữa 2 cổ tay
    dy = y1 - y2  # Nếu dy > 0: tay 1 thấp hơn tay 2 (xoay phải), dy < 0: tay 1 cao hơn tay 2 (xoay trái)
    dx = x2 - x1  # Khoảng cách ngang giữa 2 tay

    # Chuẩn hóa góc, giả sử góc tối đa là 45 độ
    max_angle = 90
    max_dy = image_height * 1.0  # Giới hạn tối đa chênh lệch để tính góc

    angle = int(np.clip((dy / max_dy) * max_angle, -max_angle, max_angle))

    if angle < 0:
        angle_left = abs(angle)
        angle_right = 0
    elif angle > 0:
        angle_left = 0
        angle_right = abs(angle)
    else:
        angle_left = 0
        angle_right = 0

    return angle_left, angle_right

# Mở camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở camera")
    exit()

mode = 0  # Khởi tạo mode mặc định là 0 (không có bàn tay)
last_speed = 0  # Thêm biến lưu tốc độ cuối cùng
prev_mode = 0   # Lưu lại mode trước đó

while True:
    # Đọc khung hình từ camera
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc khung hình từ camera")
        break
    
    # Chuyển đổi màu sắc từ BGR sang RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Xử lý khung hình với MediaPipe Hands
    results = hands.process(frame_rgb)
    
    # Vẽ landmarks nếu phát hiện bàn tay
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
    

    # Kích thước gốc
    cam_h, cam_w, _ = frame.shape # 480x640x3

    # Tính kích thước canvas mới
    canvas_h = int(cam_h / 0.7)  # Tổng chiều cao = cam_h / 0.7
    canvas_w = cam_w

    # Resize khung camera cho vừa 70% trên
    frame_resized = cv2.resize(frame, (canvas_w, int(canvas_h * 0.7)))

    # Tạo canvas trắng
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # Đặt khung camera lên 70% trên cùng
    canvas[0:int(canvas_h * 0.7), :, :] = frame_resized

    # Xử lý MediaPipe trên khung hình đã resize
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Lấy thông tin trạng thái
    hand_status = "Khong co ban tay"
    car_status = "Xe dung"
    speed_status = "Toc do: 0"
    angle_status = "Goc lai trai: 0 | phai: 0"
    num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
    current_mode = detect_hand_mode(num_hands)
    mode_buffer.append(current_mode)
    mode = Counter(mode_buffer).most_common(1)[0][0]
    if results.multi_hand_landmarks:
        if mode == 1:
            hand_landmarks = results.multi_hand_landmarks[0]
            car_status = "Dieu khien toc do"
            num_fingers = count_fingers(hand_landmarks)
            finger_buffer.append(num_fingers)
            stable_fingers = Counter(finger_buffer).most_common(1)[0][0]
            speed_status = f"Toc do: {stable_fingers}"
            hand_status = f"1 tay - {stable_fingers} ngon"
            angle_left, angle_right = get_steering_angle_wheel_style(hand_landmarks, hand_landmarks, cam_w, cam_h)
            angle_status = f"Goc lai trai: 0 | phai: 0"
            last_speed = stable_fingers
        elif mode == 2 and num_hands == 2:
            hand_landmarks_1 = results.multi_hand_landmarks[0]
            hand_landmarks_2 = results.multi_hand_landmarks[1]
            car_status = "Dieu khien goc lai"
            angle_left, angle_right = get_steering_angle_wheel_style(hand_landmarks_1, hand_landmarks_2, cam_w, cam_h)
            angle_status = f"Goc lai trai: {angle_left} | phai: {angle_right}"
            hand_status = "2 tay"
            speed_status = f"Toc do: {last_speed}"
        # else:
        #     car_status = "Xe dung"
        #     hand_status = "Khong xac dinh"
        #     speed_status = "Toc do: 0"
        #     angle_status = "Goc lai trai: 0 | phai: 0"
    else:
        mode = 0
        car_status = "Xe dung"
        speed_status = "Toc do: 0"
        angle_status = "Goc lai trai: 0 | phai: 0"
    prev_mode = mode  # Cập nhật mode trước đó

    # Vẽ khung chia ô cho phần trạng thái ở dưới
    box_height = 50
    box_margin = 10
    
    y_base = int(canvas_h * 0.7) + 40
        
    # Ô trạng thái chế độ xe (vàng nhạt)
    cv2.rectangle(canvas, (10, y_base - box_height//2), (int(canvas_w*0.45)-10, y_base + 2*box_height + box_margin), (255, 245, 170), -1)

    # Ô tốc độ/góc lái (xanh nhạt)
    cv2.rectangle(canvas, (10, y_base + 40 - box_height//2), (int(canvas_w*0.45)-10, y_base + 40 + box_height//2), (120, 230, 255), -1)

    # Ô trạng thái tay (xanh nhạt)
    cv2.rectangle(canvas, (int(canvas_w*0.5), y_base - box_height//2), (canvas_w-10, y_base + 2*box_height + box_margin), (190, 190, 255), -1)

    # Tính lại chiều cao phần trạng thái dưới
    status_top = int(canvas_h * 0.7)
    status_bottom = canvas_h - 10  # trừ 10 để sát mép dưới

    # Ô trạng thái xe (trái) phủ kín chiều cao phần dưới
    cv2.rectangle(canvas, (10, status_top + 10), (int(canvas_w*0.48)-10, status_bottom), (173, 255, 255), -1)  # xanh nhạt

    # Ô trạng thái tay (phải) phủ kín chiều cao phần dưới
    cv2.rectangle(canvas, (int(canvas_w*0.52), status_top + 10), (canvas_w-10, status_bottom), (255, 192, 203), -1)  # hồng nhạt

    # Vẽ trạng thái xe (bên trái)
    cv2.putText(canvas, car_status, (20, status_top + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 80, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, speed_status, (20, status_top + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, f"Last speed: {last_speed}", (20, status_top + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3, cv2.LINE_AA)  # <-- Dời lên ngay dưới tốc độ
    cv2.putText(canvas, angle_status, (20, status_top + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 3, cv2.LINE_AA)

    # Vẽ trạng thái tay (bên phải)
    hand_x = int(canvas_w * 0.52) + 30
    cv2.putText(canvas, "Trang thai tay:", (hand_x, status_top + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 0, 180), 3, cv2.LINE_AA)
    cv2.putText(canvas, hand_status, (hand_x, status_top + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 0, 180), 3, cv2.LINE_AA)

    # Hiển thị
    cv2.imshow("Hand Tracking", canvas)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
