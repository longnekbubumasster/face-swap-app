import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import insightface
import numpy as np
from insightface.app import FaceAnalysis

from model_assets import FACE_MODEL, MODEL_DIR, ensure_swapper_model


PROVIDERS = ["CPUExecutionProvider"]
MAX_VIDEO_SECONDS = max(1, int(os.getenv("MAX_VIDEO_SECONDS", "10")))
OUTPUT_DIR = Path(tempfile.gettempdir()) / "face-swap-app"
MODEL_LOCK = threading.Lock()


def _cleanup_old_outputs(max_age_seconds: int = 6 * 60 * 60) -> None:
    """Remove only this app's old temporary video files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - max_age_seconds
    for path in OUTPUT_DIR.glob("face-swap-*"):
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


@lru_cache(maxsize=1)
def load_models():
    """Load the CPU models once, on the first processing request."""
    swapper_path = ensure_swapper_model()
    analyser = FaceAnalysis(
        name=FACE_MODEL,
        root=str(MODEL_DIR),
        allowed_modules=["detection", "recognition"],
        providers=PROVIDERS,
    )
    analyser.prepare(ctx_id=-1, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model(
        str(swapper_path),
        download=False,
        providers=PROVIDERS,
    )
    return analyser, swapper


def _first_face(image_bgr: np.ndarray, role: str):
    analyser, _ = load_models()
    faces = analyser.get(image_bgr)
    if not faces:
        raise gr.Error(f"Không tìm thấy khuôn mặt trong {role}.")
    return faces[0]


def _sharpen(image_bgr: np.ndarray) -> np.ndarray:
    """A lightweight alternative to the large GFPGAN/PyTorch stack."""
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(image_bgr, 1.35, blurred, -0.35, 0)


def swap_face_image(target_rgb, source_rgb, enhance=True):
    if target_rgb is None or source_rgb is None:
        raise gr.Error("Vui lòng tải đủ ảnh đích và ảnh khuôn mặt nguồn.")

    target_bgr = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR)
    source_bgr = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)

    with MODEL_LOCK:
        analyser, swapper = load_models()
        source_face = _first_face(source_bgr, "ảnh nguồn")
        target_faces = analyser.get(target_bgr)
        if not target_faces:
            raise gr.Error("Không tìm thấy khuôn mặt trong ảnh đích.")

        result = target_bgr.copy()
        for target_face in target_faces:
            result = swapper.get(result, target_face, source_face, paste_back=True)

        if enhance:
            result = _sharpen(result)

    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def _browser_compatible_video(input_path: Path) -> Path:
    """Use H.264 when ffmpeg is available so browsers can play the result."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return input_path

    output_path = input_path.with_name(f"{input_path.stem}-web.mp4")
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, timeout=180)
        input_path.unlink(missing_ok=True)
        return output_path
    except (OSError, subprocess.SubprocessError):
        output_path.unlink(missing_ok=True)
        return input_path


def swap_face_video(target_video_path, source_rgb, enhance=True, progress=gr.Progress()):
    if not target_video_path or source_rgb is None:
        raise gr.Error("Vui lòng tải đủ video đích và ảnh khuôn mặt nguồn.")

    _cleanup_old_outputs()
    source_bgr = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)
    capture = cv2.VideoCapture(str(target_video_path))
    if not capture.isOpened():
        raise gr.Error("Không thể đọc video đã tải lên.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise gr.Error("Video không có kích thước hợp lệ.")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = max(1, int(fps * MAX_VIDEO_SECONDS))
    frames_to_process = min(total_frames, max_frames) if total_frames > 0 else max_frames
    raw_output = OUTPUT_DIR / f"face-swap-{uuid.uuid4().hex}.mp4"
    writer = cv2.VideoWriter(
        str(raw_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise gr.Error("Không thể tạo video kết quả.")

    processed = 0
    try:
        with MODEL_LOCK:
            analyser, swapper = load_models()
            source_face = _first_face(source_bgr, "ảnh nguồn")
            while processed < frames_to_process:
                ok, frame = capture.read()
                if not ok:
                    break

                result = frame
                for target_face in analyser.get(frame):
                    result = swapper.get(result, target_face, source_face, paste_back=True)
                if enhance:
                    result = _sharpen(result)

                writer.write(result)
                processed += 1
                if processed == 1 or processed % 5 == 0:
                    progress(processed / frames_to_process, desc="Đang xử lý video")
    finally:
        capture.release()
        writer.release()

    if processed == 0:
        raw_output.unlink(missing_ok=True)
        raise gr.Error("Video không có khung hình hợp lệ.")

    progress(1, desc="Đang chuẩn bị video để phát trên web")
    return str(_browser_compatible_video(raw_output))


CSS = """
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
.hero { text-align: center; margin-bottom: 1rem; }
.notice { color: #475569; text-align: center; }
"""


with gr.Blocks(theme=gr.themes.Soft(primary_hue="violet"), css=CSS, title="AI Face Swap Pro") as demo:
    gr.Markdown(
        """
        <div class="hero">
          <h1>🎨 AI Face Swap Pro</h1>
          <p>Ghép khuôn mặt cho ảnh và video ngay trên trình duyệt.</p>
        </div>
        <p class="notice">Chỉ sử dụng ảnh/video của bạn hoặc nội dung bạn được phép chỉnh sửa.</p>
        """
    )

    with gr.Tabs():
        with gr.TabItem("🖼️ Ghép mặt ảnh"):
            with gr.Row():
                image_target = gr.Image(type="numpy", label="Ảnh cần thay mặt")
                image_source = gr.Image(type="numpy", label="Ảnh khuôn mặt nguồn")
            image_enhance = gr.Checkbox(value=True, label="Làm nét nhẹ sau khi ghép")
            image_button = gr.Button("🚀 Ghép mặt ảnh", variant="primary")
            image_output = gr.Image(type="numpy", label="Kết quả", format="png")
            image_button.click(
                swap_face_image,
                inputs=[image_target, image_source, image_enhance],
                outputs=image_output,
            )

        with gr.TabItem("🎥 Ghép mặt video"):
            gr.Markdown(f"Video được xử lý tối đa **{MAX_VIDEO_SECONDS} giây** để tránh quá tải máy chủ.")
            with gr.Row():
                video_target = gr.Video(label="Video cần thay mặt")
                video_source = gr.Image(type="numpy", label="Ảnh khuôn mặt nguồn")
            video_enhance = gr.Checkbox(value=True, label="Làm nét nhẹ sau khi ghép")
            video_button = gr.Button("🚀 Ghép mặt video", variant="primary")
            video_output = gr.Video(label="Kết quả", format="mp4")
            video_button.click(
                swap_face_video,
                inputs=[video_target, video_source, video_enhance],
                outputs=video_output,
            )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=4).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
        show_error=True,
    )
