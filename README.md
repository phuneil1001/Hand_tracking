# Hand Tracking Car Control

## Mô tả dự án

Đây là dự án điều khiển ô tô mô hình bằng cử chỉ tay sử dụng camera máy tính. Hệ thống nhận diện số ngón tay và trạng thái bàn tay (xòe/khép) để điều khiển tốc độ và góc lái của xe mô hình. Dự án sử dụng các thư viện chính: OpenCV, MediaPipe, NumPy, Streamlit và streamlit-webrtc để xây dựng giao diện web trực quan.

- **Đầu vào:** Camera máy tính (webcam)
- **Đầu ra:** Điều khiển xe mô hình (hiển thị trạng thái trên giao diện)
- **Các cử chỉ nhận diện:** Số ngón tay, trạng thái xòe/khép bàn tay

## Tính năng

- Nhận diện bàn tay và số ngón tay đang duỗi (không tính ngón cái)
- Phân biệt trạng thái xòe tay (điều khiển tốc độ) và khép tay (điều khiển góc lái)
- Hiển thị trạng thái điều khiển (tốc độ, góc lái, trạng thái tay) trực tiếp trên giao diện web
- Giao diện web realtime sử dụng Streamlit và streamlit-webrtc

## Hướng dẫn cài đặt

1. **Cài đặt Python >= 3.8**
2. **Cài đặt các thư viện cần thiết:**
3. **Chạy ứng dụng web:**

## Triển khai trên Streamlit Cloud

Dự án có thể được triển khai trực tuyến trên: https://handtracking-qswp5nhzkdueriomdd2xwi.streamlit.app/

## Cấu trúc thư mục

- `app.py`: Ứng dụng web Streamlit nhận diện cử chỉ tay
- `main.py`, `One_hand.py`, `Two_hand.py`: Các file thử nghiệm nhận diện tay với OpenCV/MediaPipe
- `requirements.txt`: Danh sách thư viện cần thiết
- `DockerFile`: Hỗ trợ chạy ứng dụng trong Docker (có GPU)
- `test.py`: File kiểm tra môi trường và thư viện

## Ghi chú

- Để sử dụng tính năng nhận diện realtime, cần cấp quyền truy cập camera cho trình duyệt.
- Nếu chạy trong Docker, cần cấu hình đúng thiết bị camera và GPU (nếu có).
- Font tiếng Việt có thể cần bổ sung nếu muốn hiển thị Unicode trên giao diện OpenCV.
