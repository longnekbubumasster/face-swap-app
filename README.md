# AI Face Swap Pro

Ứng dụng Gradio ghép khuôn mặt vào ảnh hoặc video, được cấu hình để chạy trên Render bằng CPU.

## Deploy lên Render

> **Lưu ý chi phí:** model AI này không phù hợp với gói Free 512 MB. File `render.yaml` dùng gói `standard` (1 CPU, 2 GB RAM). Render sẽ hiển thị chi phí trước khi bạn xác nhận tạo dịch vụ.

1. Push mã nguồn lên GitHub. Không thêm file `inswapper_128.onnx` vì file này lớn hơn giới hạn của GitHub và đã được `.gitignore` bỏ qua.
2. Trong Render, chọn **New → Blueprint**.
3. Kết nối repository và chọn file `render.yaml`.
4. Xác nhận tạo dịch vụ. Bước build sẽ tự cài package và tải model; lần đầu có thể mất khá lâu.
5. Khi trạng thái chuyển sang **Live**, mở URL `onrender.com` do Render cung cấp.

Render tự triển khai lại sau mỗi lần push lên nhánh đã kết nối.

## Chạy local

Yêu cầu Python 3.11 và ít nhất khoảng 2 GB RAM trống.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy==1.26.4 Cython==3.0.11
python -m pip install --no-build-isolation --no-deps insightface==0.7.3
python -m pip install -r requirements.txt
python -m scripts.download_models
python app.py
```

Mở `http://localhost:7860`. Có thể đổi giới hạn video bằng biến môi trường `MAX_VIDEO_SECONDS` (mặc định 10 giây).

## Cấu hình

- `PORT`: cổng web; Render tự cung cấp biến này.
- `MODEL_DIR`: thư mục chứa model, mặc định `.models`.
- `FACE_MODEL`: model nhận diện InsightFace, mặc định `buffalo_s` để tiết kiệm RAM.
- `MAX_VIDEO_SECONDS`: số giây video tối đa được xử lý, mặc định `10`.
- `INSWAPPER_MODEL_PATH`: đường dẫn model `inswapper_128.onnx` tùy chỉnh.

Chỉ sử dụng ảnh/video của bạn hoặc nội dung bạn được phép chỉnh sửa.
