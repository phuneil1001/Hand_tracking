import cv2
import mediapipe as mp

# Khởi tạo bộ phát hiện tay
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # Lấy vị trí đầu ngón trỏ (landmark 8) và gốc tay (landmark 0)
            x_tip = handLms.landmark[8].x
            x_base = handLms.landmark[0].x

            # Điều khiển đơn giản bằng cách phân tích vị trí x của đầu ngón trỏ
            if x_tip < x_base - 0.1:
                print("Rẽ trái")
            elif x_tip > x_base + 0.1:
                print("Rẽ phải")
            else:
                print("Đi thẳng")

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
