import onnxruntime as ort
import numpy as np
from PIL import Image
import yaml
import argparse
from scipy.special import softmax
import os

# ==== Đọc tham số từ command line ====
parser = argparse.ArgumentParser(description="ONNX Classification Inference")
parser.add_argument("-c", "--config", type=str, required=True, help="Path to config file")
parser.add_argument("-o", "--override", action="append", help="Override config options")
args = parser.parse_args()

# ==== Load config từ YAML ====
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# ==== Ghi đè nếu có -o ====
if args.override:
    for item in args.override:
        key, value = item.split("=", 1)
        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value

# ==== Lấy các tham số chính từ config ====
image_path = config["Global"]["infer_imgs"]
model_path = r"D:\Capcha\PaddleClas\output2\best_model\model.onnx"
label_file = config["PostProcess"]["Topk"]["class_id_map_file"]
topk = int(config["PostProcess"]["Topk"].get("topk", 5))

# ==== Load nhãn ====
with open(label_file, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]

# ==== Hàm resize giữ tỉ lệ chiều ngắn nhất ====
def resize_short(img, resize_short=256):
    width, height = img.size
    if width < height:
        new_width = resize_short
        new_height = int(resize_short * height / width)
    else:
        new_height = resize_short
        new_width = int(resize_short * width / height)
    return img.resize((new_width, new_height), Image.BILINEAR)

# ==== Hàm crop chính giữa ====
def center_crop(img, size):
    width, height = img.size
    left = (width - size) // 2
    top = (height - size) // 2
    return img.crop((left, top, left + size, top + size))

# ==== Tiền xử lý hình ảnh ====
def preprocess_image(image_path, input_size=224):
    img = Image.open(image_path).convert("RGB")
    img = resize_short(img, resize_short=256)
    img = center_crop(img, size=input_size)

    img = np.array(img).astype('float32') * 0.00392157  # = /255.0

    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
    img = (img - mean) / std

    img = img.transpose((2, 0, 1))  # [C, H, W]
    img = np.expand_dims(img, axis=0)  # [1, C, H, W]
    return img.astype(np.float32)

# ==== Load ONNX model ====
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ==== In thông tin shape model ====
print(f"Input shape: {session.get_inputs()[0].shape}")
print(f"Output shape: {session.get_outputs()[0].shape}")

# ==== Tiền xử lý và dự đoán ====
input_data = preprocess_image(image_path)
outputs = session.run([output_name], {input_name: input_data})[0]

probs = softmax(outputs[0], axis=-1)
top_indices = np.argsort(probs)[::-1][:topk]
top_probs = probs[top_indices]

# ==== CHUẨN HÓA & LÀM SẮC NÉT ====
sharpening_factor = 5  # thử tăng lên 10 nếu muốn top1 càng rõ
top_probs = top_probs ** sharpening_factor
top_probs = top_probs / np.sum(top_probs)

# ==== In kết quả ====
print(f"\nTop {topk} predictions (sharpened & normalized):")
for idx, prob in zip(top_indices, top_probs):
    label = labels[idx] if idx < len(labels) else f"Class {idx}"
    print(f"Class: {label}, Confidence: {prob:.4f}")


# ==== (Tùy chọn) Lưu kết quả vào file ====
if "SavePreLabel" in config.get("PostProcess", {}):
    save_dir = config["PostProcess"]["SavePreLabel"].get("save_dir", "./output_labels")
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(save_dir, base_name + ".txt")
    with open(save_path, "w", encoding="utf-8") as f:
        for idx, prob in zip(top_indices, top_probs):
            label = labels[idx] if idx < len(labels) else f"Class {idx}"
            f.write(f"{label}\t{prob:.4f}\n")
    print(f"\nSaved result to: {save_path}")
