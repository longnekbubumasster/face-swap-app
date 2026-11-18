import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from gfpgan import GFPGANer
import streamlit as st
from PIL import Image

st.set_page_config(page_title="AI Face Swap Pro", page_icon="🎨", layout="centered")

@st.cache_resource
def load_models():
    app_face = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app_face.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True)
    restorer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
        upscale=1,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None
    )
    return app_face, swapper, restorer

app_face, swapper, restorer = load_models()

st.title("🎨 AI Face Swap Pro")
st.write("Ghép mặt sắc nét, tối ưu cho iPhone.")

tab1, tab2 = st.tabs(["🖼️ Ghép mặt Ảnh", "🎥 Ghép mặt Video"])

with tab1:
    st.subheader("Ghép mặt vào Ảnh")
    target_file = st.file_uploader("1. Chọn Ảnh Cần Thay Mặt", type=["jpg", "jpeg", "png"], key="target_img")
    source_file = st.file_uploader("2. Chọn Ảnh Khuôn Mặt Gốc", type=["jpg", "jpeg", "png"], key="source_img")

    if st.button("🚀 Tiến Hành Ghép Mặt Ảnh", type="primary"):
        if target_file and source_file:
            with st.spinner("Đang xử lý ghép mặt và làm nét..."):
                target_bytes = np.asarray(bytearray(target_file.read()), dtype=np.uint8)
                target_img = cv2.imdecode(target_bytes, 1)

                source_bytes = np.asarray(bytearray(source_file.read()), dtype=np.uint8)
                source_img = cv2.imdecode(source_bytes, 1)

                source_faces = app_face.get(source_img)
                if len(source_faces) == 0:
                    st.error("Không tìm thấy khuôn mặt trong ảnh nguồn!")
                else:
                    source_face = source_faces[0]
                    target_faces = app_face.get(target_img)
                    if len(target_faces) == 0:
                        st.error("Không tìm thấy khuôn mặt nào trong ảnh đích!")
                    else:
                        res_img = target_img.copy()
                        for face in target_faces:
                            res_img = swapper.get(res_img, face, source_face, paste_back=True)

                        _, _, enhanced_img = restorer.enhance(
                            res_img, has_aligned=False, only_center_face=False, paste_back=True
                        )

                        output_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
                        st.image(output_rgb, caption="Kết quả thành công! Chạm giữ ảnh để lưu về iPhone.", use_column_width=True)
        else:
            st.warning("Vui lòng tải lên đầy đủ cả 2 ảnh!")

with tab2:
    st.subheader("Ghép mặt vào Video")
    video_target = st.file_uploader("1. Chọn Video từ Thư viện", type=["mp4", "mov"], key="target_vid")
    video_source = st.file_uploader("2. Chọn Ảnh Khuôn Mặt Gốc", type=["jpg", "jpeg", "png"], key="source_vid_img")

    if st.button("🚀 Tiến Hành Ghép Mặt Video", type="primary"):
        if video_target and video_source:
            with st.spinner("Đang xử lý video (giới hạn ngắn để chạy mượt)..."):
                video_path = "temp_video.mp4"
                with open(video_path, "wb") as f:
                    f.write(video_target.read())

                source_bytes = np.asarray(bytearray(video_source.read()), dtype=np.uint8)
                source_img = cv2.imdecode(source_bytes, 1)
                source_faces = app_face.get(source_img)

                if len(source_faces) == 0:
                    st.error("Không tìm thấy khuôn mặt trong ảnh nguồn!")
                else:
                    source_face = source_faces[0]
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    max_frames = int(fps * 10)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    target_frames = min(total_frames, max_frames)

                    output_path = "output_swapped_pro.mp4"
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

                        _, _, enhanced_frame = restorer.enhance(
                            res_frame, has_aligned=False, only_center_face=False, paste_back=True
                        )
                        out.write(enhanced_frame)
                        frame_count += 1

                    cap.release()
                    out.release()

                    st.success("Xử lý video thành công!")
                    st.video(output_path)
        else:
            st.warning("Vui lòng tải lên video và ảnh khuôn mặt!")