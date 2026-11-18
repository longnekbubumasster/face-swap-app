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
VIDEO_MAX_FPS = max(1.0, float(os.getenv("VIDEO_MAX_FPS", "6")))
IMAGE_MAX_DIM = max(1024, int(os.getenv("IMAGE_MAX_DIM", "2560")))
SOURCE_MAX_DIM = max(640, int(os.getenv("SOURCE_MAX_DIM", "1600")))
VIDEO_MAX_DIM = max(640, int(os.getenv("VIDEO_MAX_DIM", "1280")))
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "true").lower() in {"1", "true", "yes", "on"}
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
    analyser.prepare(ctx_id=-1, det_thresh=0.4, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model(
        str(swapper_path),
        download=False,
        providers=PROVIDERS,
    )
    return analyser, swapper


def _resize_long_side(image: np.ndarray, max_dim: int, min_dim: int = 0) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    scale = 1.0
    if longest > max_dim:
        scale = max_dim / longest
    elif min_dim and longest < min_dim:
        scale = min_dim / longest
    if abs(scale - 1.0) < 0.001:
        return np.ascontiguousarray(image)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)


def _face_crop(image_bgr: np.ndarray, face, margin: float = 0.45) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in face.bbox]
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + margin * 2)
    left = max(0, int(center_x - side / 2))
    top = max(0, int(center_y - side / 2))
    right = min(width, int(center_x + side / 2))
    bottom = min(height, int(center_y + side / 2))
    return np.ascontiguousarray(image_bgr[top:bottom, left:right])


def _face_quality(image_bgr: np.ndarray, face) -> float:
    crop = _face_crop(image_bgr, face, margin=0.05)
    if crop.size == 0:
        return -1.0
    gray = cv2.cvtColor(_resize_long_side(crop, 320), cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    x1, y1, x2, y2 = [float(value) for value in face.bbox]
    face_side = max(x2 - x1, y2 - y1)
    size_score = min(1.0, face_side / 280.0)
    sharpness_score = min(1.0, sharpness / 220.0)
    confidence = float(getattr(face, "det_score", 0.5) or 0.5)
    orientation_score = 0.5
    keypoints = getattr(face, "kps", None)
    if keypoints is not None and len(keypoints) >= 5:
        left_eye, right_eye, nose, left_mouth, right_mouth = keypoints[:5]
        eye_angle = abs(float(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))))
        eye_angle = min(eye_angle, abs(180 - eye_angle))
        level_score = max(0.0, 1.0 - eye_angle / 45.0)
        eye_y = float((left_eye[1] + right_eye[1]) / 2)
        mouth_y = float((left_mouth[1] + right_mouth[1]) / 2)
        upright_score = 1.0 if mouth_y > nose[1] > eye_y else 0.0
        orientation_score = (level_score + upright_score) / 2
    return size_score * 0.35 + sharpness_score * 0.25 + confidence * 0.15 + orientation_score * 0.25


def _best_source_face(image_bgr: np.ndarray, role: str):
    analyser, _ = load_models()
    search_image = _resize_long_side(image_bgr, SOURCE_MAX_DIM, min_dim=640)
    best = None
    for turns in range(4):
        candidate = np.ascontiguousarray(np.rot90(search_image, turns))
        faces = analyser.get(candidate)
        for face in faces:
            score = _face_quality(candidate, face)
            if best is None or score > best[0]:
                best = (score, candidate, face)
    if best is None:
        raise gr.Error(f"Không tìm thấy khuôn mặt rõ trong {role}. Hãy dùng ảnh chính diện, đủ sáng.")

    _, oriented_image, original_face = best
    crop = _face_crop(oriented_image, original_face)
    refined_image = _resize_long_side(crop, 960, min_dim=640)
    refined_faces = analyser.get(refined_image)
    if refined_faces:
        refined_face = max(refined_faces, key=lambda face: _face_quality(refined_image, face))
        return refined_face, crop
    return original_face, crop


def _target_faces_with_orientation(analyser, image_bgr: np.ndarray):
    for turns in range(4):
        candidate = np.ascontiguousarray(np.rot90(image_bgr, turns))
        faces = analyser.get(candidate)
        if faces:
            return candidate, faces, turns
    raise gr.Error("Không tìm thấy khuôn mặt rõ trong ảnh đích. Hãy dùng ảnh đủ sáng.")


def _sharpen(image_bgr: np.ndarray) -> np.ndarray:
    """A lightweight alternative to the large GFPGAN/PyTorch stack."""
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(image_bgr, 1.35, blurred, -0.35, 0)


def swap_face_image(target_rgb, source_rgb, enhance=True, progress=gr.Progress()):
    if target_rgb is None or source_rgb is None:
        raise gr.Error("Vui lòng tải đủ ảnh đích và ảnh khuôn mặt nguồn.")

    progress(0, desc="Đang nạp model AI")
    target_bgr = _resize_long_side(cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR), IMAGE_MAX_DIM, min_dim=512)
    source_bgr = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)

    with MODEL_LOCK:
        analyser, swapper = load_models()
        progress(0.3, desc="Đang nhận diện khuôn mặt")
        source_face, source_crop = _best_source_face(source_bgr, "ảnh nguồn")
        oriented_target, target_faces, turns = _target_faces_with_orientation(analyser, target_bgr)

        progress(0.6, desc=f"Đang ghép {len(target_faces)} khuôn mặt")
        result = oriented_target.copy()
        for target_face in target_faces:
            result = swapper.get(result, target_face, source_face, paste_back=True)

        if enhance:
            result = _sharpen(result)
        if turns:
            result = np.ascontiguousarray(np.rot90(result, -turns))

    progress(1, desc="Hoàn tất")
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB), cv2.cvtColor(source_crop, cv2.COLOR_BGR2RGB)


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

    video_scale = min(1.0, VIDEO_MAX_DIM / max(width, height))
    output_width = max(2, int(round(width * video_scale / 2) * 2))
    output_height = max(2, int(round(height * video_scale / 2) * 2))

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    max_input_frames = max(1, int(fps * MAX_VIDEO_SECONDS))
    input_frames_to_read = min(total_frames, max_input_frames) if total_frames > 0 else max_input_frames
    # Face detection is the expensive part. Sampling to a web-friendly FPS keeps
    # the original duration while making CPU-only Render instances respond sooner.
    frame_step = max(1, int(round(fps / VIDEO_MAX_FPS)))
    output_fps = fps / frame_step
    frames_to_process = max(1, (input_frames_to_read + frame_step - 1) // frame_step)
    raw_output = OUTPUT_DIR / f"face-swap-{uuid.uuid4().hex}.mp4"
    writer = cv2.VideoWriter(
        str(raw_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        capture.release()
        raise gr.Error("Không thể tạo video kết quả.")

    processed = 0
    frames_read = 0
    try:
        progress(0, desc="Đang nạp model AI")
        with MODEL_LOCK:
            analyser, swapper = load_models()
            source_face, source_crop = _best_source_face(source_bgr, "ảnh nguồn")
            while frames_read < input_frames_to_read:
                ok, frame = capture.read()
                if not ok:
                    break
                current_frame = frames_read
                frames_read += 1
                if current_frame % frame_step != 0:
                    continue

                if (frame.shape[1], frame.shape[0]) != (output_width, output_height):
                    frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)

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
    return str(_browser_compatible_video(raw_output)), cv2.cvtColor(source_crop, cv2.COLOR_BGR2RGB)


def _preload_models() -> None:
    try:
        with MODEL_LOCK:
            load_models()
        print("AI models preloaded and ready.", flush=True)
    except Exception as error:
        print(f"Model preload failed: {error}", flush=True)


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
            with gr.Row():
                image_output = gr.Image(type="numpy", label="Kết quả", format="png")
                image_source_preview = gr.Image(type="numpy", label="Khuôn mặt nguồn được tự động chọn")
            image_button.click(
                swap_face_image,
                inputs=[image_target, image_source, image_enhance],
                outputs=[image_output, image_source_preview],
            )

        with gr.TabItem("🎥 Ghép mặt video"):
            gr.Markdown(f"Video được xử lý tối đa **{MAX_VIDEO_SECONDS} giây** để tránh quá tải máy chủ.")
            with gr.Row():
                video_target = gr.Video(label="Video cần thay mặt")
                video_source = gr.Image(type="numpy", label="Ảnh khuôn mặt nguồn")
            video_enhance = gr.Checkbox(value=True, label="Làm nét nhẹ sau khi ghép")
            video_button = gr.Button("🚀 Ghép mặt video", variant="primary")
            with gr.Row():
                video_output = gr.Video(label="Kết quả", format="mp4")
                video_source_preview = gr.Image(type="numpy", label="Khuôn mặt nguồn được tự động chọn")
            video_button.click(
                swap_face_video,
                inputs=[video_target, video_source, video_enhance],
                outputs=[video_output, video_source_preview],
            )


if __name__ == "__main__":
    if PRELOAD_MODELS:
        threading.Thread(target=_preload_models, name="model-preloader", daemon=True).start()
    demo.queue(default_concurrency_limit=1, max_size=4).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
        show_error=True,
    )
