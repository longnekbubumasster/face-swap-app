import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from gfpgan import GFPGANer
import gradio as gr

# Khởi tạo mô hình AI
print("Đang tải mô hình AI và bộ lọc làm nét...")
app_face = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app_face.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True)

# Khởi tạo mô hình làm nét GFPGAN (giúp mặt giống thật, sắc sảo hơn)
restorer = GFPGANer(
    model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

# 1. Xử lý ghép mặt vào Video
def process_face_swap_video(video_file, image_file):
    if video_file is None or image_file is None:
        return None, "Vui lòng chọn đầy đủ video và ảnh chân dung!"

    video_path = video_file.name
    image_path = image_file.name
    output_path = "output_swapped_pro.mp4"

    source_img = cv2.imread(image_path)
    source_faces = app_face.get(source_img)
    if len(source_faces) == 0:
        return None, "Không tìm thấy khuôn mặt trong ảnh nguồn!"
    source_face = source_faces[0]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    max_frames = int(fps * 10) # Giới hạn 10 giây để chạy mượt
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frames = min(total_frames, max_frames)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened() and frame_count < target_frames:
        ret, frame = cap.read()
        if not ret:
            break

        target_faces = app_face.get(frame)
        res_frame = frame.copy()
        
        for face in target_faces:
            res_frame = swapper.get(res_frame, face, source_face, paste_back=True)

        # Làm nét khuôn mặt bằng GFPGAN
        _, _, enhanced_frame = restorer.enhance(
            res_frame, 
            has_aligned=False, 
            only_center_face=False, 
            paste_back=True
        )

        out.write(enhanced_frame)
        frame_count += 1

    cap.release()
    out.release()
    
    return output_path, "Xử lý thành công! Nhấn tải xuống bên dưới để lưu vào iPhone."

# 2. Xử lý ghép mặt vào Ảnh
def process_face_swap_image(target_image_file, source_image_file):
    if target_image_file is None or source_image_file is None:
        return None, "Vui lòng chọn đầy đủ ảnh đích và ảnh nguồn!"

    target_path = target_image_file.name
    source_path = source_image_file.name
    output_path = "output_swapped_pro.jpg"

    source_img = cv2.imread(source_path)
    source_faces = app_face.get(source_img)
    if len(source_faces) == 0:
        return None, "Không tìm thấy khuôn mặt trong ảnh nguồn!"
    source_face = source_faces[0]

    target_img = cv2.imread(target_path)
    target_faces = app_face.get(target_img)
    if len(target_faces) == 0:
        return None, "Không tìm thấy khuôn mặt nào trong ảnh đích!"

    res_img = target_img.copy()
    for face in target_faces:
        res_img = swapper.get(res_img, face, source_face, paste_back=True)

    # Làm nét siêu sắc sảo cho ảnh
    _, _, enhanced_img = restorer.enhance(
        res_img, 
        has_aligned=False, 
        only_center_face=False, 
        paste_back=True
    )

    cv2.imwrite(output_path, enhanced_img)
    return output_path, "Xử lý thành công! Nhấn vào ảnh kết quả để lưu vào Thư viện iPhone."

# Giao diện Web Gradio tối ưu cho iPhone 14 Pro Max
custom_css = """
body { background-color: #f7f7f8; }
.gradio-container { max-width: 600px !important; margin: auto; padding: 15px; }
.primary-btn { background: linear-gradient(135deg, #007aff, #5856d6) !important; color: white !important; font-size: 16px !important; font-weight: bold; border-radius: 12px !important; padding: 12px !important; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Default()) as demo:
    gr.Markdown("# 🎨 AI Face Swap Pro (iPhone Optimized)")
    gr.Markdown("Chọn ảnh/video trực tiếp từ Thư viện iPhone, xử lý nét căng & dễ dàng lưu về máy.")

    with gr.Tabs():
        with gr.TabItem("🖼️ Ghép mặt Ảnh"):
            with gr.Column():
                img_target = gr.Image(label="1. Chọn Ảnh Cần Thay Mặt", type="filepath", sources=["upload"])
                img_source = gr.Image(label="2. Chọn Ảnh Khuôn Mặt Gốc", type="filepath", sources=["upload"])
                img_btn = gr.Button("🚀 Tiến Hành Ghép Mặt (Siêu Nét)", elem_classes="primary-btn")
                img_status = gr.Textbox(label="Trạng thái", interactive=False)
                # Trả về dạng Image component giúp người dùng Safari bấm giữ hoặc chạm để lưu trực tiếp vào Thư viện ảnh
                img_output = gr.Image(label="📥 Kết Quả (Chạm hoặc giữ ảnh để lưu về iPhone)", type="filepath")

            img_btn.click(fn=process_face_swap_image, inputs=[img_target, img_source], outputs=[img_output, img_status])

        with gr.TabItem("🎥 Ghép mặt Video"):
            with gr.Column():
                vid_input = gr.Video(label="1. Chọn Video từ Thư viện", sources=["upload"])
                vid_source_img = gr.Image(label="2. Chọn Ảnh Khuôn Mặt Gốc", type="filepath", sources=["upload"])
                vid_btn = gr.Button("🚀 Tiến Hành Ghép Video", elem_classes="primary-btn")
                vid_status = gr.Textbox(label="Trạng thái", interactive=False)
                vid_output = gr.Video(label="📥 Kết Quả Video & Tải Xuống")

            vid_btn.click(fn=process_face_swap_video, inputs=[vid_input, vid_source_img], outputs=[vid_output, vid_status])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)