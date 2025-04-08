import yaml
import paddle
from paddle.io import DataLoader, RandomSampler, SequentialSampler
from paddle.vision.transforms import Compose, Normalize, Resize, RandomCrop, RandomHorizontalFlip
from paddle.io import Dataset

# Đọc cấu hình từ file YAML
with open(r"D:\Capcha\PaddleClas\deploy\configs\config.yaml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# Lấy thông tin từ cấu hình
train_image_root = config['DataLoader']['Train']['dataset']['image_root']
train_label_path = config['DataLoader']['Train']['dataset']['cls_label_path']
eval_image_root = config['DataLoader']['Eval']['dataset']['image_root']
eval_label_path = config['DataLoader']['Eval']['dataset']['cls_label_path']

# Khởi tạo dataset (dữ liệu huấn luyện và đánh giá) -- Đây chỉ là ví dụ, cần tùy chỉnh cho phù hợp với dự án của bạn
class CustomDataset(Dataset):
    def __init__(self, image_root, label_path, transform=None):
        self.image_root = image_root
        self.label_path = label_path
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        # Đây là nơi bạn đọc các file ảnh và nhãn từ thư mục
        # Ví dụ bạn có thể sử dụng glob để lấy danh sách ảnh trong thư mục
        return [(image_path, label) for image_path, label in zip(self.image_root, self.label_path)]
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.data)

# Áp dụng các phép biến đổi cho dữ liệu
train_transforms = Compose([
    Resize(size=(224, 224)),
    RandomCrop(size=(224, 224)),
    RandomHorizontalFlip(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = Compose([
    Resize(size=(224, 224)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Khởi tạo dataset
train_dataset = CustomDataset(image_root=train_image_root, label_path=train_label_path, transform=train_transforms)
eval_dataset = CustomDataset(image_root=eval_image_root, label_path=eval_label_path, transform=eval_transforms)

# Dùng RandomSampler cho huấn luyện
train_sampler = RandomSampler(train_dataset)
train_loader = DataLoader(dataset=train_dataset, batch_size=16, sampler=train_sampler)

# Dùng BatchSampler với shuffle=False cho đánh giá
eval_sampler = BatchSampler(eval_dataset, batch_size=16, shuffle=False)
eval_loader = DataLoader(dataset=eval_dataset, batch_sampler=eval_sampler)

# Khởi tạo mô hình (ví dụ EfficientNetB0)
model = paddle.vision.models.EfficientNetB0(num_classes=config['Arch']['class_num'], pretrained=config['Arch']['pretrained'])

# Khởi tạo optimizer và loss function
optimizer = paddle.optimizer.Adam(learning_rate=config['Global']['learning_rate'], parameters=model.parameters())
criterion = paddle.nn.CrossEntropyLoss()

# Huấn luyện mô hình
for epoch in range(config['Global']['epochs']):
    model.train()  # Chuyển mô hình sang chế độ huấn luyện
    for batch_id, (images, labels) in enumerate(train_loader):
        # Di chuyển dữ liệu và mô hình sang GPU nếu có
        if paddle.get_device() == "gpu":
            images, labels = images.cuda(), labels.cuda()

        # Tiến hành tính toán loss và cập nhật weights
        optimizer.clear_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if batch_id % config['Global']['log_interval'] == 0:
            print(f"Epoch {epoch}, Batch {batch_id}, Loss {loss.numpy()}")

    # Đánh giá mô hình sau mỗi epoch
    model.eval()
    correct = 0
    total = 0
    with paddle.no_grad():
        for batch_id, (images, labels) in enumerate(eval_loader):
            logits = model(images)
            predicted = paddle.argmax(logits, axis=1)
            correct += paddle.sum(predicted == labels).numpy()
            total += labels.shape[0]
        
        accuracy = correct / total
        print(f"Epoch {epoch}, Accuracy {accuracy}")
