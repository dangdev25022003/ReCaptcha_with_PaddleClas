giải mã capcha là một công viêc khá là phức tạp với nhiều bà toán khác nhau cho từng loại captcha. Trong dự án của mình tôi muốn ứng dụng AI vào việc giải quyết các bài toán liên quan đến Captcha. Cụ thể bài toán đầu tiên tôi ứng dụng việc này là về các bài toán lựa chọn đối tượng dựa vào text. RC3 với bài toán này tôi dùng đến Paddleclas của PaddlePaddleddaay là 1 framework thịnh hành và có độ chính xác cao cũng như tổng hợp nhiều loại model tốt nhất hiện nay.
Về dữ liệu chúng tôi sẽ tìm và thu thập từ những API của các trang. Về model ở đây tôi sử dụng model EfficientB0, Vì sao sử dụng model này cho bài toán. Thứ nhất tối ưu cho việc tích hợp vào một hệ thống có thể cho sau này. Thức 2 tuy model nhẹ hơn rất nhiều so với các VS khác nhưng độ chính xác lại đạt ngữơng ấn tựng ngang ngữa với các VS khác và điển vượt trội như nêu trên là về thời gian Huấn luyện cũng như dự đoán là rất nhanh.
Cách chạy thì hoàn toàn theo như cấu trúc của Paddle:
Đầu tiên dự liệu:
Tất nhiên là với bài toán class việc chuẩn bị data chuẩn xác là quan trọng. Chúng tôi đã thu thập được hơn 5k data cho 16 class hay xuất hiện nhất. Tiếp đến bước xử lý data: khi thu thâp data dữ liệu rất tạp nham yêu cầu một số chọn lọc ở đây tôi sẽ áp dụng việc train một model nhỏ với lượng data chính xác mỗi class gồm 200 ảnh với YOLO. Sau đó sẽ dùng model này dự đoán trên toàn bộ tập dữ liệu. Tiếp đến với dữ liệu đã được phân nhãn trong 16 folder với độ chính xác đạt hơn 80% chúng tôi sử dụng một số phương pháp gồm so khớp hình ảnh để loại bỏ ảnh bị giống hệt nhau (Khó tránh khỏi khi clone data), và một số phương pháp khác bao gồm cả thủ công để chọn lọc ra lượng data chính xác nhất cho các class. Vấn đề kế tiếp là cần tăng cường để hạn chế mất cân bằng dữ liệu. Ở đây phương pháp tôi đưa ra là 50/50 nghĩa là dưới ngưỡng 50 % data của class đó thiếu so với class có nhiều data nhất thì sẽ tăng cường x2 đến x3 cho ảnh và trường hợp còn lại giữ nguyên. Sau khi đã xử lý data cuối cùng sẽ cố xấp xĩ 1000 ảnh cho mỗi class.
![image](https://github.com/dangdev25022003/ReCaptcha_with_PaddleClas/blob/main/%E1%BA%A2nh%20ch%E1%BB%A5p%20m%C3%A0n%20h%C3%ACnh%202025-04-08%20144640.png)
Cấu trúc data huấn luyện:
```dataset/
├── train/
│   ├── class_0/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   ├── class_1/
│   │   ├── img003.jpg
│   │   ├── img004.jpg
│   └── ...
│   └── class_15/
│       ├── imgXYZ.jpg
│
├── val/
│   ├── class_0/
│   ├── class_1/
│   └── ...
│
└── test/
    ├── img001.jpg
    ├── img002.jpg
    └── ...
```
Ngoài ra bạn cần tạo các file.txt gồm
```
    ├── Train_list.txt  chứa các đường dẫn là class của ảnh: car/aug_101.jpg 12
    ├── Val_list.txt  chứa các đường dẫn là class của ảnh: parkingmeters/aug_1000.jpg 11
    ├── Lables.txt #chứa các tên class theo dòng
```

Huấn luyện:
tìm đến file config của mô hình bạn muốn chọn ở đây mình dùng EfficientNetB0.yaml 
sau khi điền đầy đủ thông tin cần thiết kế đến bạn truyền các đường đãn cần thiết vào câu lệnh sau 
```
python tools/train.py -c ./ppcls/configs/quick_start/EfficientB0.yaml -o Arch.pretrained=True
```
Bạn cần đọc qua MD của paddle để cài đặt một số thứ khác nếu gặp lỗi.
Tiếp tới là export model đã train:
```
python tools/export_model.py
    -c ./ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml \
    -o Global.pretrained_model=./cls_pretrain/ResNet50_vd_pretrained \
    -o Global.save_inference_dir=./deploy/models/class_ResNet50_vd_ImageNet_infer
```
Sau đó là convert sang onnx nếu muốn. 
```
    paddle2onnx --model_dir=./models/ResNet50_vd_infer/ \
    --model_filename=inference.pdmodel \
    --params_filename=inference.pdiparams \
    --save_file=./models/ResNet50_vd_infer/inference.onnx \
    --opset_version=10 \
    --enable_onnx_checker=True
```
Dự đoán model thường:
```
python predict_cls.py -c D:\Capcha\PaddleClas\deploy\configs\inference_cls.yaml  -o Global.infer_imgs=D:\data_class_img\test_out\images.jpg -o Global.inference_model_dir=D:\Capcha\PaddleClas\output2\inference_model -o PostProcess.Topk.class_id_map_file=D:\Capcha\PaddleClas\deploy\configs\labels.txt
```
Dự đoán model ONNX:
File dự đoán:
```
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
```
Câu lệnh dự đoán:
```
python D:\Capcha\PaddleClas\inf_mode.py -c D:\Capcha\PaddleClas\deploy\configs\inference_cls.yaml -o Global.infer_imgs=D:\data_class_img\test_out\istockphoto-1340161559-612x612.jpg -o PostProcess.Topk.class_id_map_file=D:\Capcha\PaddleClas\deploy\configs\labels.txt
```
