import os
import time
import cv2
import gdown
import torch
import functools
import numpy as np
import gradio as gr
import torch.nn as nn
import torch.nn.functional as F

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


# ============================================================
# NEXTGEN VISION AI — Gradio Webcam Demo
# Use this ONLY for webcam/camera capture.
#
# Pipeline:
# Webcam/Image Capture -> DCP + ResNet Dehazing -> YOLO Detection -> AR Overlay
# ============================================================

DEHAZE_MODEL_PATH = "remove_hazy_model_256x256.pth"
DEHAZE_GDRIVE_ID = "1ji3x-KO19X2yGpT7oaUIpJ5DiCgQg8xS"
YOLO_MODEL_NAME = "yolov8n.pt"


# ============================================================
# MODEL DOWNLOAD
# ============================================================

def download_dehaze_model_if_needed():
    if os.path.exists(DEHAZE_MODEL_PATH):
        return True, "Model already exists."

    try:
        url = f"https://drive.google.com/uc?id={DEHAZE_GDRIVE_ID}"
        gdown.download(url, DEHAZE_MODEL_PATH, quiet=False)

        if os.path.exists(DEHAZE_MODEL_PATH):
            return True, "Model downloaded successfully."

        return False, "Download finished, but model file was not found."
    except Exception as e:
        return False, str(e)


# ============================================================
# DEHAZING ARCHITECTURE
# ============================================================

class GuidedFilter(nn.Module):
    def __init__(self, r=40, eps=1e-3):
        super(GuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        self.boxfilter = nn.AvgPool2d(
            kernel_size=2 * self.r + 1,
            stride=1,
            padding=self.r,
        )

    def forward(self, I, p):
        N = self.boxfilter(torch.ones(p.size(), device=p.device, dtype=p.dtype))

        mean_I = self.boxfilter(I) / N
        mean_p = self.boxfilter(p) / N
        mean_Ip = self.boxfilter(I * p) / N
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self.boxfilter(I * I) / N
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I

        mean_a = self.boxfilter(a) / N
        mean_b = self.boxfilter(b) / N

        return mean_a * I + mean_b


class DCPDehazeGenerator(nn.Module):
    def __init__(self, win_size=15, r=40, eps=1e-3):
        super(DCPDehazeGenerator, self).__init__()
        self.guided_filter = GuidedFilter(r=r, eps=eps)
        self.neighborhood_size = win_size
        self.omega = 0.95

    def get_dark_channel(self, img, w):
        img, _ = torch.min(img, dim=1)
        img = torch.unsqueeze(img, dim=1)

        pad_size = int(np.floor(w / 2))
        if w % 2 == 0:
            pads = [pad_size, pad_size - 1, pad_size, pad_size - 1]
        else:
            pads = [pad_size, pad_size, pad_size, pad_size]

        img_min = F.pad(img, pads, mode="replicate")
        dark_img = -F.max_pool2d(-img_min, kernel_size=w, stride=1)

        return dark_img

    def atmospheric_light(self, img, dark_img):
        num, chl, height, width = img.shape
        top_num = max(int(0.001 * height * width), 1)

        A = torch.zeros(num, chl, 1, 1, device=img.device, dtype=img.dtype)

        for num_id in range(num):
            cur_img = img[num_id]
            cur_dark_img = dark_img[num_id, 0]

            _, indices = cur_dark_img.reshape(height * width).sort(descending=True)

            for chl_id in range(chl):
                img_slice = cur_img[chl_id].reshape(height * width)
                A[num_id, chl_id, 0, 0] = torch.mean(img_slice[indices[:top_num]])

        return A

    def forward(self, x):
        # x expected in [-1, 1]
        guidance = (
            0.2989 * x[:, 0, :, :] +
            0.5870 * x[:, 1, :, :] +
            0.1140 * x[:, 2, :, :]
        )

        guidance = torch.unsqueeze((guidance + 1) / 2, dim=1)
        img_patch = (x + 1) / 2

        num, chl, height, width = img_patch.shape

        dark_img = self.get_dark_channel(img_patch, self.neighborhood_size)
        A = self.atmospheric_light(img_patch, dark_img)

        map_A = A.repeat(1, 1, height, width).clamp(min=1e-6)

        trans_raw = 1 - self.omega * self.get_dark_channel(
            img_patch / map_A,
            self.neighborhood_size,
        )
        trans_raw = trans_raw.clamp(min=0.05, max=1.0)

        T_DCP = self.guided_filter(guidance, trans_raw).clamp(min=0.05, max=1.0)

        J_DCP = (img_patch - map_A) / T_DCP.repeat(1, 3, 1, 1) + map_A

        return J_DCP.clamp(0, 1)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        block = []

        for i in range(2):
            if padding_type == "reflect":
                block += [nn.ReflectionPad2d(1)]
                padding = 0
            elif padding_type == "replicate":
                block += [nn.ReplicationPad2d(1)]
                padding = 0
            else:
                padding = 1

            block += [
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=padding, bias=use_bias),
                norm_layer(dim),
            ]

            if i == 0:
                block += [nn.ReLU(True)]
                if use_dropout:
                    block += [nn.Dropout(0.5)]

        self.conv_block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        n_blocks=9,
        padding_type="reflect",
    ):
        super().__init__()

        use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        for i in range(2):
            mult = 2 ** i
            model += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        mult = 4

        for _ in range(n_blocks):
            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        for i in range(2):
            mult = 2 ** (2 - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0, bias=use_bias),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return torch.clamp(self.model(x), min=-1, max=1)


# ============================================================
# LOAD MODELS
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DCP_MODEL = None
RESNET_MODEL = None
YOLO_MODEL = None


def load_all_models():
    global DCP_MODEL, RESNET_MODEL, YOLO_MODEL

    ok, msg = download_dehaze_model_if_needed()
    if not ok:
        raise RuntimeError(msg)

    if DCP_MODEL is None:
        DCP_MODEL = DCPDehazeGenerator().to(DEVICE).eval()

    if RESNET_MODEL is None:
        RESNET_MODEL = ResnetGenerator(
            input_nc=3,
            output_nc=3,
            norm_layer=nn.InstanceNorm2d,
        ).to(DEVICE)

        ckpt = torch.load(DEHAZE_MODEL_PATH, map_location=DEVICE)

        if isinstance(ckpt, dict):
            key = next(
                (k for k in ["params", "state_dict", "model", "net_g", "generator"] if k in ckpt),
                None,
            )
            state_dict = ckpt[key] if key else ckpt
        else:
            state_dict = ckpt

        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        RESNET_MODEL.load_state_dict(state_dict, strict=False)
        RESNET_MODEL.eval()

    if YOLO_AVAILABLE and YOLO_MODEL is None:
        YOLO_MODEL = YOLO(YOLO_MODEL_NAME)

    return f"Models loaded on {DEVICE}"


# Load once during startup
STARTUP_STATUS = load_all_models()


# ============================================================
# PROCESSING FUNCTIONS
# ============================================================

DRIVING_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
}


def bgr_to_tensor_minus1_to_1(img_bgr, size):
    img_bgr = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_CUBIC)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor * 2.0 - 1.0


def tensor_0_1_to_bgr(tensor, original_hw):
    out = tensor.squeeze(0).detach().cpu().clamp(0, 1).numpy()
    out = out.transpose(1, 2, 0)

    out_bgr = cv2.cvtColor(
        (out * 255.0).round().astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )

    h, w = original_hw
    return cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_CUBIC)


def dehaze_bgr(img_bgr, strength=1.0, inference_size=192, dcp_only=False):
    h, w = img_bgr.shape[:2]
    x = bgr_to_tensor_minus1_to_1(img_bgr, inference_size).to(DEVICE)

    with torch.no_grad():
        dcp_out = DCP_MODEL(x)

        if dcp_only:
            refined = dcp_out
        else:
            refined = (RESNET_MODEL(dcp_out) + 1.0) / 2.0

        result = tensor_0_1_to_bgr(refined, (h, w))

    if strength < 1.0:
        result = cv2.addWeighted(img_bgr, 1.0 - strength, result, strength, 0)

    return result


def detect_bgr(img_bgr, conf_threshold=0.35, driving_only=True):
    if YOLO_MODEL is None:
        return img_bgr, []

    results = YOLO_MODEL(img_bgr, conf=conf_threshold, verbose=False)
    result = results[0]

    annotated = img_bgr.copy()
    detections = []

    if result.boxes is None:
        return annotated, detections

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = YOLO_MODEL.names[cls_id]

        if driving_only and name not in DRIVING_CLASSES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        detections.append(
            {
                "class": name,
                "confidence": round(conf, 2),
                "box": [int(x1), int(y1), int(x2), int(y2)],
            }
        )

        color = (0, 255, 150)
        if name in ["person", "bicycle", "motorcycle"]:
            color = (0, 90, 255)
        elif name in ["traffic light", "stop sign"]:
            color = (0, 212, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{name.upper()} {conf:.2f}"
        label_y = max(y1 - 8, 20)

        font_scale = max(0.42, min(0.60, annotated.shape[1] / 1100))
        thickness = 1 if annotated.shape[1] < 700 else 2

        cv2.rectangle(
            annotated,
            (x1, label_y - 18),
            (x1 + max(90, int(len(label) * 9)), label_y + 5),
            color,
            -1,
        )

        cv2.putText(
            annotated,
            label,
            (x1 + 5, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (2, 6, 23),
            thickness,
            cv2.LINE_AA,
        )

    return annotated, detections


def draw_hud(img_bgr, mode, detections_count, processing_time):
    out = img_bgr.copy()
    h, w = out.shape[:2]

    overlay = out.copy()
    cv2.rectangle(overlay, (10, 10), (min(w - 10, 460), 66), (2, 6, 23), -1)
    out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)

    cv2.putText(
        out,
        f"NEXTGEN VISION AI | {mode}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 180),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        f"Objects: {detections_count} | Time: {processing_time:.2f}s",
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 212, 255),
        1,
        cv2.LINE_AA,
    )

    return out


def process_camera_frame(
    img_rgb,
    strength,
    inference_size,
    conf_threshold,
    enable_dehazing,
    enable_detection,
    driving_only,
    dcp_only,
):
    if img_rgb is None:
        return None, None, "Please capture or upload an image first."

    start = time.time()

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if enable_dehazing:
        dehazed_bgr = dehaze_bgr(
            img_bgr,
            strength=float(strength),
            inference_size=int(inference_size),
            dcp_only=bool(dcp_only),
        )
    else:
        dehazed_bgr = img_bgr.copy()

    detections = []

    if enable_detection:
        final_bgr, detections = detect_bgr(
            dehazed_bgr,
            conf_threshold=float(conf_threshold),
            driving_only=bool(driving_only),
        )
    else:
        final_bgr = dehazed_bgr

    elapsed = time.time() - start

    final_bgr = draw_hud(
        final_bgr,
        mode="WEBCAM CAPTURE",
        detections_count=len(detections),
        processing_time=elapsed,
    )

    dehazed_rgb = cv2.cvtColor(dehazed_bgr, cv2.COLOR_BGR2RGB)
    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)

    summary = {
        "processing_time_seconds": round(elapsed, 3),
        "objects_detected": len(detections),
        "detections": detections,
        "device": str(DEVICE),
        "dehazing": bool(enable_dehazing),
        "object_detection": bool(enable_detection),
    }

    return dehazed_rgb, final_rgb, summary


# ============================================================
# GRADIO UI
# ============================================================

custom_css = """
body {
    background: #020617 !important;
}

.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(34, 211, 238, 0.16), transparent 34%),
        radial-gradient(circle at top right, rgba(52, 211, 153, 0.10), transparent 31%),
        linear-gradient(135deg, #020617 0%, #050b18 55%, #020617 100%) !important;
    color: #e5f4ff !important;
    font-family: Inter, system-ui, sans-serif !important;
}

.panel, .block {
    background: rgba(15, 23, 42, 0.78) !important;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    border-radius: 20px !important;
}

h1, h2, h3 {
    color: #e5f4ff !important;
}

button.primary {
    background: linear-gradient(135deg, #22d3ee, #38bdf8) !important;
    color: #020617 !important;
    font-weight: 900 !important;
    border-radius: 14px !important;
}

button.secondary {
    border-radius: 14px !important;
}
"""

with gr.Blocks(
    title="NEXTGEN VISION AI — Webcam Demo",
    css=custom_css,
    theme=gr.themes.Soft(primary_hue="cyan", neutral_hue="slate"),
) as demo:
    gr.Markdown(
        """
        # 👁️ NEXTGEN VISION AI — Webcam Capture Demo

        Use this page only for the **webcam/camera part** of your presentation.  
        Capture a frame from the camera, then run the full pipeline:

        **Webcam Capture → Dehazing → YOLO Object Detection → AR-style Output**
        """
    )

    gr.Markdown(f"**Startup status:** `{STARTUP_STATUS}`")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Camera / Upload Input",
                sources=["webcam", "upload"],
                type="numpy",
                height=420,
            )

            with gr.Accordion("Controls", open=True):
                strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.05,
                    label="Enhancement Strength",
                )

                inference_size = gr.Dropdown(
                    choices=[128, 192, 256],
                    value=192,
                    label="Dehazing Inference Size",
                )

                conf_threshold = gr.Slider(
                    minimum=0.10,
                    maximum=0.90,
                    value=0.35,
                    step=0.05,
                    label="YOLO Confidence Threshold",
                )

                enable_dehazing = gr.Checkbox(value=True, label="Enable Dehazing")
                enable_detection = gr.Checkbox(value=True, label="Enable YOLO Detection")
                driving_only = gr.Checkbox(value=True, label="Driving-Related Classes Only")
                dcp_only = gr.Checkbox(value=False, label="DCP Only Mode")

            run_btn = gr.Button("🚀 Process Camera Frame", variant="primary")

        with gr.Column(scale=1):
            dehazed_output = gr.Image(
                label="Dehazed Output",
                type="numpy",
                height=350,
            )

            final_output = gr.Image(
                label="Final Output with Object Detection",
                type="numpy",
                height=350,
            )

            summary_output = gr.JSON(label="Processing Summary")

    run_btn.click(
        fn=process_camera_frame,
        inputs=[
            input_image,
            strength,
            inference_size,
            conf_threshold,
            enable_dehazing,
            enable_detection,
            driving_only,
            dcp_only,
        ],
        outputs=[
            dehazed_output,
            final_output,
            summary_output,
        ],
    )

    gr.Markdown(
        """
        ### Presentation line you can say
        “For the camera prototype, the system captures a live frame from the camera and immediately applies the same AI pipeline:
        dehazing for visibility enhancement, followed by YOLO-based hazard detection with AR-style bounding boxes.”
        """
    )


if __name__ == "__main__":
    demo.launch()
