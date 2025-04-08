from paddleclas import PaddleClas

# Cách 1: Chỉ định trực tiếp file model
model = PaddleClas(
    model_file="D:/Capcha/PaddleClass/output2/inference_model/inference.pdmodel",
    params_file="D:/Capcha/PaddleClass/output2/inference_model/inference.pdiparams",
    config_file="D:/Capcha/PaddleClass/output2/inference_model/inference.yml"  # Nếu có
)

# Cách 2: Chỉ định thư mục model (tự động tìm file)
model = PaddleClas(
    inference_model_dir="D:/Capcha/PaddleClass/output2/inference_model/"
)

# Dự đoán ảnh
result = model.predict(
    input_data="D:/data_class_img/2.jpg",  # Thay bằng đường dẫn ảnh của bạn
    print_probs=True,  # Hiển thị xác suất
    topk=3  # Hiển thị top 3 kết quả
)

print(result)