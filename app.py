# Dự án điều khiển ô tô mô hình bằng cử chỉ tay
# Đầu vào: camera máy tính
# Đầu ra: điều khiển ô tô mô hình
# Đây là file test để kiểm tra các cử chỉ tay có thể nhận diện được hay không
# Các cử chỉ tay được nhận diện: số ngón tay
# Thư viện: OpenCV, MediaPipe, NumPy

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter
import math

# Buffer để làm mượt trạng thái
MODE_BUFFER_SIZE = 10
mode_buffer = deque([0]*MODE_BUFFER_SIZE, maxlen=MODE_BUFFER_SIZE)
FINGER_BUFFER_SIZE = 30
finger_buffer = deque([0]*FINGER_BUFFER_SIZE, maxlen=FINGER_BUFFER_SIZE)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def detect_hand_mode(hand_landmarks, image_width, image_height):
    if hand_landmarks is None:
        return 0
    landmarks = hand_landmarks.landmark
    finger_tips = [8, 12, 16, 20]
    finger_coords = [(landmarks[i].x * image_width, landmarks[i].y * image_height) for i in finger_tips]
    total_dist = 0
    count = 0
    for i in range(len(finger_coords)):
        for j in range(i+1, len(finger_coords)):
            dx = finger_coords[i][0] - finger_coords[j][0]
            dy = finger_coords[i][1] - finger_coords[j][1]
            dist = (dx**2 + dy**2)**0.5
            total_dist += dist
            count += 1
    avg_dist = total_dist / count
    wrist = (landmarks[0].x * image_width, landmarks[0].y * image_height)
    middle_finger = (landmarks[12].x * image_width, landmarks[12].y * image_height)
    palm_length = ((wrist[0] - middle_finger[0])**2 + (wrist[1] - middle_finger[1])**2)**0.5
    if palm_length == 0:
        return 0
    norm_avg_dist = avg_dist / palm_length
    threshold = 0.27
    if norm_avg_dist > threshold:
        return 1  # Xòe tay
    else:
        return 2  # Khép tay

def count_fingers(hand_landmarks):
    lm = hand_landmarks.landmark
    fingers = 0
    tip_ids = [8, 12, 16, 20]
    for tip_id in tip_ids:
        if lm[tip_id].y < lm[tip_id - 2].y:
            fingers += 1
    return fingers

def get_steering_angles(hand_landmarks, image_width, image_height):
    if hand_landmarks is None:
        return 0, 0
    landmarks = hand_landmarks.landmark
    x0, y0 = landmarks[0].x * image_width, landmarks[0].y * image_height
    x12, y12 = landmarks[12].x * image_width, landmarks[12].y * image_height
    dx = x12 - x0
    dy = y0 - y12
    angle_rad = math.atan2(dx, dy)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_right = abs(int(angle_deg))
        angle_left = 0
    else:
        angle_right = 0
        angle_left = int(angle_deg)
    return angle_left, angle_right

class HandVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.last_speed = 0
        self.prev_mode = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        cam_h, cam_w, _ = img.shape

        # Trạng thái mặc định
        mode = 0
        car_status = "Xe dung"
        speed_status = "Toc do: 0"
        angle_status = "Goc lai trai: 0 | phai: 0"
        hand_status = "Khong co ban tay"

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            current_mode = detect_hand_mode(hand_landmarks, cam_w, cam_h)
            mode_buffer.append(current_mode)
            mode = Counter(mode_buffer).most_common(1)[0][0]
            if mode == 1:
                car_status = "Dieu khien toc do"
                num_fingers = count_fingers(hand_landmarks)
                finger_buffer.append(num_fingers)
                stable_fingers = Counter(finger_buffer).most_common(1)[0][0]
                speed_status = f"Toc do: {stable_fingers}"
                hand_status = f"Xoe tay - {stable_fingers} ngon"
                angle_left, angle_right = get_steering_angles(hand_landmarks, cam_w, cam_h)
                angle_status = f"Goc lai trai: 0 | phai: 0"
                self.last_speed = stable_fingers
            elif mode == 2:
                car_status = "Dieu khien goc lai"
                angle_left, angle_right = get_steering_angles(hand_landmarks, cam_w, cam_h)
                angle_status = f"Goc lai trai: {angle_left} | phai: {angle_right}"
                hand_status = "Khep tay"
                if self.prev_mode == 1 or self.prev_mode == 2:
                    speed_status = f"Toc do: {self.last_speed}"
                elif self.prev_mode == 0:
                    speed_status = f"Toc do: {self.last_speed}"
                else:
                    speed_status = "Toc do: 0"
            else:
                car_status = "Xe dung"
                hand_status = "Khong xac dinh"
                speed_status = "Toc do: 0"
                angle_status = "Goc lai trai: 0 | phai: 0"
        else:
            mode = 0
            car_status = "Xe dung"
            speed_status = "Toc do: 0"
            angle_status = "Goc lai trai: 0 | phai: 0"
        self.prev_mode = mode

        # Ghi trạng thái lên ảnh
        cv2.rectangle(img, (0, 0), (img.shape[1], 90), (255, 255, 255), -1)
        cv2.putText(img, car_status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 80, 0), 2, cv2.LINE_AA)
        cv2.putText(img, speed_status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2, cv2.LINE_AA)
        cv2.putText(img, angle_status, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, hand_status, (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 0, 180), 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Nhận diện cử chỉ tay điều khiển xe mô hình (MediaPipe + Streamlit)")

webrtc_streamer(
    key="hand-detect",
    video_processor_factory=HandVideoProcessor,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": True, "audio": False},
)
