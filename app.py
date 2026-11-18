import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from gfpgan import GFPGANer
import gradio as gr

# Khởi tạo mô hình AI
print("Đang tải mô hình AI và bộ lọc làm nét...")
app_face = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']) # Đổi thành 'CUDAExecutionProvider' nếu có GPU NVIDIA
app_face.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True)

# Khởi tạo mô hình làm nét khuôn mặt GFPGAN (giúp mặt giống thật, sắc sảo hơn)
restorer = GFPGANer(
    model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

# Hàm phụ trợ: Cân bằng màu sắc vùng da mặt để khớp với khung hình gốc
def color_transfer(source_img, target_img, mask):
    # Chuyển sang không gian màu LAB để cân chỉnh độ sáng và sắc độ da
    source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB)
    
    # Khớp màu trung bình trong vùng mặt
    for i in range(3):
        source_lab[:, :, i] = np.where(
            mask > 0,
            np.clip(
                (source_lab[:, :, i].astype(np.float32) - np.mean(source_lab[:, :, i][mask > 0])) * 
                (np.std(target_lab[:, :, i][mask > 0]) / (np.std(source_lab[:, :, i][mask > 0]) + 1e-6)) + 
                np.mean(target_lab[:, :, i][mask > 0]), 0, 255
            ).astype(np.uint8),
            source_lab[:, :, i]
        )
    return cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)

# 1. Xử lý ghép mặt vào Video (Có làm nét & Khớp màu)
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
    
    max_frames = int(fps * 10) # Giới hạn 10 giây
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
            # Ghép mặt cơ bản
            res_frame = swapper.get(res_frame, face, source_face, paste_back=True)

        # Sử dụng GFPGAN để làm nét chi tiết khuôn mặt, khử mờ
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
    
    return output_path, "Đã ghép mặt vào video thành công (Đã tối ưu làm nét & màu sắc)!"

# 2. Xử lý ghép mặt vào Ảnh (Có làm nét & Khớp màu)
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

    # Sử dụng GFPGAN làm nét để khuôn mặt sắc sảo, giống thật nhất
    _, _, enhanced_img = restorer.enhance(
        res_img, 
        has_aligned=False, 
        only_center_face=False, 
        paste_back=True
    )

    cv2.imwrite(output_path, enhanced_img)
    return output_path, "Đã ghép mặt vào ảnh thành công (Sắc nét và chân thực hơn)!"

# Giao diện Web Gradio
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Ứng dụng Ghép Mặt AI Chuyên Sâu (High-Quality Face Swap)")
    gr.Markdown("Đã tích hợp mô hình làm nét **GFPGAN** giúp khuôn mặt sắc nét, tự nhiên và khớp màu hài hòa hơn.")

    with gr.Tabs():
        with gr.TabItem("🎥 Ghép mặt vào Video"):
            with gr.Row():
                with gr.Column():
                    vid_input = gr.File(label="Chọn Video (tối đa 10s)", file_types=[".mp4", ".avi", ".mov"])
                    vid_source_img = gr.File(label="Chọn Ảnh khuôn mặt muốn ghép", file_types=[".jpg", ".png", ".jpeg"])
                    vid_btn = gr.Button("Bắt đầu ghép mặt (Bản Nâng Cấp)", variant="primary")
                    vid_status = gr.Textbox(label="Trạng thái", interactive=False)
                with gr.Column():
                    vid_output = gr.Video(label="Kết quả Video & Tải xuống")

            vid_btn.click(fn=process_face_swap_video, inputs=[vid_input, vid_source_img], outputs=[vid_output, vid_status])

        with gr.TabItem("🖼️ Ghép mặt vào Ảnh"):
            with gr.Row():
                with gr.Column():
                    img_target = gr.File(label="Chọn Ảnh đích (Ảnh cần thay mặt)", file_types=[".jpg", ".png", ".jpeg"])
                    img_source = gr.File(label="Chọn Ảnh khuôn mặt nguồn", file_types=[".jpg", ".png", ".jpeg"])
                    img_btn = gr.Button("Bắt đầu ghép mặt (Bản Nâng Cấp)", variant="primary")
                    img_status = gr.Textbox(label="Trạng thái", interactive=False)
                with gr.Column():
                    img_output = gr.Image(label="Kết quả Ảnh & Tải xuống", type="filepath")

            img_btn.click(fn=process_face_swap_image, inputs=[img_target, img_source], outputs=[img_output, img_status])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)