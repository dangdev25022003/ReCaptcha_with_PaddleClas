import os
# import random

# Đường dẫn thư mục
train_dir = r"D:\data_class_img\224x224_train_2_4_v2\train"
val_dir = r"D:\data_class_img\224x224_train_2_4_v2\val"
output_dir = r"D:\Capcha\PaddleClas\deploy\configs"

# Hàm tạo file label đúng định dạng
def create_proper_label(input_dir, output_file):
    classes = sorted(os.listdir(input_dir))
    with open(os.path.join(output_dir, output_file), 'w') as f:
        for class_idx, class_name in enumerate(classes):
            for img in os.listdir(os.path.join(input_dir, class_name)):
                f.write(f"{class_name}/{img} {class_idx}\n")  # Thêm class_idx

# Tạo file mới
create_proper_label(train_dir, "train_list.txt")
create_proper_label(val_dir, "val_list.txt")
# import os
# import re

# # Đường dẫn thư mục
# train_dir = r"D:\data_class_img\224x224_train_2_4_v2\train"
# val_dir = r"D:\data_class_img\224x224_train_2_4_v2\val"
# output_dir = r"D:\Capcha\PaddleClas\deploy\configs"

# def sanitize_filename(filename):
#     """Chuẩn hóa tên file: thay khoảng trắng bằng gạch dưới, bỏ ký tự đặc biệt"""
#     # Thay khoảng trắng bằng gạch dưới
#     filename = filename.replace(" ", "_")
#     # Loại bỏ các ký tự đặc biệt khác (giữ lại dấu gạch dưới, dấu chấm và chữ cái/số)
#     filename = re.sub(r'[^\w.-]', '', filename)
#     return filename

# def process_images_and_create_label(input_dir, output_file):
#     classes = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    
#     with open(os.path.join(output_dir, output_file), 'w', encoding='utf-8') as f:
#         for class_idx, class_name in enumerate(classes):
#             class_path = os.path.join(input_dir, class_name)
            
#             for old_name in os.listdir(class_path):
#                 old_path = os.path.join(class_path, old_name)
                
#                 if os.path.isfile(old_path):
#                     # Lấy phần mở rộng file
#                     ext = os.path.splitext(old_name)[1].lower()
#                     if ext not in ['.jpg', '.jpeg', '.png']:
#                         continue
                    
#                     # Tạo tên mới
#                     new_name = sanitize_filename(old_name)
#                     new_path = os.path.join(class_path, new_name)
                    
#                     # Đổi tên file
#                     if old_name != new_name:
#                         os.rename(old_path, new_path)
#                         print(f"Đã đổi tên: {old_name} -> {new_name}")
                    
#                     # Ghi vào file label
#                     f.write(f"{class_name}/{new_name} {class_idx}\n")

# # Xử lý ảnh train và tạo label
# print("Đang xử lý thư mục train...")
# process_images_and_create_label(train_dir, "train_list.txt")

# # Xử lý ảnh val và tạo label
# print("\nĐang xử lý thư mục val...")
# process_images_and_create_label(val_dir, "val_list.txt")

# print("\nHoàn thành! Đã:")
# print("- Chuẩn hóa tên tất cả ảnh (thay khoảng trắng bằng gạch dưới)")
# print("- Tạo file train_list.txt và val_list.txt")