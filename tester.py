"""
Real-Time Violence Detection — ResNet-18 3D CNN (r3d_18)
Load the trained model using the Load Model button, then start the webcam.
"""

import gradio as gr
import torch
import torch.nn.functional as F
from torchvision.models.video import r3d_18
import cv2
import numpy as np
from collections import deque

# ── Custom video transforms (match training pipeline in notebook) ─────────────
class ResizeVideo:
    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        # video: (C, T, H, W)
        c, t, h, w = video.shape
        out = torch.zeros((c, t, self.size, self.size), dtype=video.dtype)
        for i in range(t):
            frame = video[:, i, :, :].unsqueeze(0)
            out[:, i, :, :] = F.interpolate(
                frame, size=(self.size, self.size), mode='bilinear', align_corners=False
            ).squeeze(0)
        return out

class NormalizeVideo:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1, 1)
        self.std  = torch.tensor(std).view(-1,  1, 1, 1)

    def __call__(self, video):
        # video: (C, T, H, W)
        return (video - self.mean) / self.std

TRANSFORM = [
    ResizeVideo(112),
    NormalizeVideo(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
]

def apply_transforms(video):
    for t in TRANSFORM:
        video = t(video)
    return video

# ── Globals ───────────────────────────────────────────────────────────────────
current_model  = None
current_device = None
frame_buffer        = deque(maxlen=16)
violence_vote_buffer = deque(maxlen=5)

VIOLENCE_THRESHOLD     = 0.85
REQUIRED_VIOLENT_VOTES = 5

# ── Model loading (button click) ──────────────────────────────────────────────
def load_r3d_model(model_path_str, device_choice):
    global current_model, current_device
    import os

    if not os.path.exists(model_path_str):
        return f"❌ File not found: {model_path_str}"

    try:
        device = torch.device(device_choice)

        # Build model with SAME fc architecture as notebook
        model = r3d_18(weights=None)
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(model.fc.in_features, 2)
        )

        checkpoint = torch.load(model_path_str, map_location=device, weights_only=True)

        # Handle both full checkpoint dict and raw state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint  # best_model_weights.pth is a raw state dict

        model.load_state_dict(state_dict)
        model.to(device).eval()

        current_model = model
        current_device = device_choice
        print(f"Model loaded from {model_path_str} on {device_choice}")
        return f"✅ Model loaded on {device_choice}"

    except Exception as e:
        current_model = None
        return f"❌ Load failed: {str(e)[:200]}"

# ── Webcam discovery ──────────────────────────────────────────────────────────
def get_available_webcams():
    available = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(str(i))
            cap.release()
    return available or ["0"]

# ── Per-frame prediction ──────────────────────────────────────────────────────
def predict_live_stream(image, cam_idx):
    global frame_buffer, violence_vote_buffer

    if image is None:
        return image

    prediction = "Waiting for frames..."
    confidence = 0.0
    color = (120, 120, 120)

    try:
        frame_buffer.append(image)   # raw RGB (H, W, 3)

        if current_model is None:
            prediction = "Load model first (click button above)"
            color = (0, 140, 255)

        elif len(frame_buffer) == 16:
            frames_np = np.stack(list(frame_buffer), axis=0)       # (T, H, W, C)
            frames_np = frames_np.transpose((3, 0, 1, 2))          # (C, T, H, W)
            video_tensor = torch.from_numpy(frames_np).float() / 255.0

            video_tensor = apply_transforms(video_tensor).unsqueeze(0)
            video_tensor = video_tensor.to(torch.device(current_device))

            with torch.no_grad():
                outputs = current_model(video_tensor)
                probs   = torch.softmax(outputs, dim=1)[0]
                pred_idx   = int(torch.argmax(probs).item())
                confidence = float(probs[pred_idx].item())

            labels = ['Non-Violent', 'Violent']
            is_confident_violent = (pred_idx == 1 and confidence >= VIOLENCE_THRESHOLD)
            violence_vote_buffer.append(is_confident_violent)
            sustained = all(violence_vote_buffer) and len(violence_vote_buffer) == REQUIRED_VIOLENT_VOTES

            if sustained:
                prediction = f"VIOLENT ({confidence:.0%})"
                color = (0, 0, 255)
            elif is_confident_violent:
                prediction = f"Possible Violent ({confidence:.0%})"
                color = (0, 140, 255)
            else:
                prediction = f"{labels[pred_idx]} ({confidence:.0%})"
                color = (0, 200, 0)

        # ── Draw overlay ──────────────────────────────────────────────────────
        overlay = image.copy()
        h, w = overlay.shape[:2]

        bar = overlay.copy()
        cv2.rectangle(bar, (0, h - 52), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(bar, 0.55, overlay, 0.45, 0, overlay)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.75, 2
        tw = cv2.getTextSize(prediction, font, scale, thick)[0][0]
        tx = max(10, (w - tw) // 2)
        cv2.putText(overlay, prediction, (tx, h - 16), font, scale, color, thick, cv2.LINE_AA)

        fill_w = int(w * len(frame_buffer) / 16)
        cv2.rectangle(overlay, (0, h - 55), (fill_w, h - 52), (80, 200, 100), -1)

        return overlay

    except Exception as e:
        err = image.copy()
        cv2.putText(err, f"Error: {str(e)[:60]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return err

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Violence Detection — R3D-18", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔴 Real-Time Violence Detection (ResNet-18 3D CNN)")

    with gr.Row():
        with gr.Column(scale=1):
            model_path_tb = gr.Textbox(
                label="Model Path (.pth)",
                value="/home/heytanix/PCL_repository/violence_classifier.pth",
                placeholder="Full path to your trained .pth file"
            )
            device_dd = gr.Dropdown(
                choices=["cpu", "cuda"],
                label="Device",
                value="cuda" if torch.cuda.is_available() else "cpu"
            )
            load_btn = gr.Button("🔄 Load Model", variant="primary")
            model_status = gr.Textbox(
                label="Model Status",
                value="Not loaded. Enter path and click Load Model.",
                interactive=False,
                lines=2
            )
            load_btn.click(fn=load_r3d_model, inputs=[model_path_tb, device_dd], outputs=model_status)

            cam_index = gr.Dropdown(
                choices=get_available_webcams(),
                label="Select Webcam",
                value=get_available_webcams()[0]
            )
            gr.Markdown("""
### Legend
🟢 **Non-Violent** — normal activity  
🟠 **Possible Violent** — single high-confidence detection  
🔴 **VIOLENT** — 5 consecutive detections ≥ 85% confidence
            """)

        with gr.Column(scale=2):
            webcam_input = gr.Image(
                sources=["webcam"],
                streaming=True,
                label="📷 Webcam Input"
            )
            output_display = gr.Image(
                label="🎯 Detection Output",
                interactive=False
            )

    webcam_input.stream(
        fn=predict_live_stream,
        inputs=[webcam_input, cam_index],
        outputs=output_display,
        stream_every=0.1,
        show_progress="hidden"
    )

if __name__ == "__main__":
    demo.launch(share=False, debug=False)