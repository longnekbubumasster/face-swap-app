# AI Face Swap Pro — chạy trên Windows

Ứng dụng ghép khuôn mặt cho ảnh và video bằng CPU, chạy hoàn toàn trên máy tính local. Ảnh và video không cần gửi lên Render.

## Cách nhanh nhất

Yêu cầu: Windows 10/11, Python 3.12 64-bit và khoảng 4 GB RAM trống.

### Lần đầu

1. Double-click `setup_local.bat`.
2. Chờ cài thư viện và tải model hoàn tất.
3. Double-click `start_local.bat`.

Trình duyệt sẽ tự mở địa chỉ `http://localhost:7860` sau khi model đã nạp xong.

### Những lần sau

Chỉ cần double-click `start_local.bat`.

## Chạy bằng PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
$env:MODEL_DIR = "$PWD\.models"
.\.venv\Scripts\python.exe -m scripts.download_models
.\.venv\Scripts\python.exe app.py
```

## Hoạt động

- Tự tìm khuôn mặt nguồn rõ nhất.
- Tự xoay và crop khuôn mặt nguồn.
- Lấy pixel thật của toàn bộ vùng mặt nguồn thay cho mặt cũ, không dùng model sinh mặt kiểu búp bê.
- Ghép tất cả khuôn mặt phát hiện được trong ảnh đích.
- Ảnh nhỏ được phóng lên để nhận diện; ảnh rất lớn được giảm về kích thước an toàn.
- Video được giảm FPS và độ phân giải khi cần để xử lý nhanh hơn trên CPU.
- Model được lưu tại `.models` và chỉ tải một lần.

## Cấu hình tùy chọn

- `PORT`: cổng local, mặc định `7860`.
- `MAX_VIDEO_SECONDS`: độ dài video tối đa, mặc định `10` giây.
- `VIDEO_MAX_FPS`: FPS xử lý tối đa, mặc định `6`.
- `VIDEO_MAX_DIM`: cạnh dài tối đa của video, mặc định `1280`.
- `IMAGE_MAX_DIM`: cạnh dài tối đa của ảnh kết quả, mặc định `2560`.
- `OPEN_BROWSER`: tự mở trình duyệt, mặc định `true`.

Chỉ sử dụng ảnh/video của bạn hoặc nội dung bạn được phép chỉnh sửa.
