import os
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import cv2
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import imghdr

# логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_valid_image(file_path):
    try:
        if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return False
        
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return False
        
        image_type = imghdr.what(file_path)
        if image_type not in ['jpeg', 'png']:
            return False
        
        img = cv2.imread(file_path)
        if img is None or img.size == 0:
            return False
        
        return True
    except Exception as e:
        logger.warning(f"Ошибка при проверке изображения {file_path}: {str(e)}")
        return False

class RMFDDataset(Dataset):
    def __init__(self, root_dir, max_samples_per_person=10):
        self.root_dir = root_dir
        self.max_samples_per_person = max_samples_per_person
        self.masked_dir = os.path.join(root_dir, 'masked')
        self.unmasked_dir = os.path.join(root_dir, 'unmasked')
        
        if not os.path.exists(self.masked_dir) or not os.path.exists(self.unmasked_dir):
            raise ValueError("Директории 'masked' и 'unmasked' должны находиться в корневой папке датасета")
        
        self.masked_files = self._get_image_files(self.masked_dir)
        self.unmasked_files = self._get_image_files(self.unmasked_dir)
        self._balance_dataset()
        
        self.masked_labels = [1] * len(self.masked_files)  # 1 для масок
        self.unmasked_labels = [0] * len(self.unmasked_files)  # 0 для лиц без масок
        self.files = self.masked_files + self.unmasked_files
        self.labels = self.masked_labels + self.unmasked_labels
        
        combined = list(zip(self.files, self.labels))
        random.shuffle(combined)
        self.files, self.labels = zip(*combined)
        
        logger.info(f"Загружено {len(self.masked_files)} изображений с масками")
        logger.info(f"Загружено {len(self.unmasked_files)} изображений без масок")
    
    def _get_image_files(self, directory):
        image_files = []
        invalid_files = []
        
        for person_dir in os.listdir(directory):
            person_path = os.path.join(directory, person_dir)
            if os.path.isdir(person_path):
                for file in os.listdir(person_path):
                    file_path = os.path.join(person_path, file)
                    if is_valid_image(file_path):
                        image_files.append(file_path)
                    else:
                        invalid_files.append(file_path)
        
        return image_files
    
    def _balance_dataset(self):
        masked_by_person = {}
        unmasked_by_person = {}
        
        for file in self.masked_files:
            person = os.path.basename(os.path.dirname(file))
            if person not in masked_by_person:
                masked_by_person[person] = []
            masked_by_person[person].append(file)
        
        for file in self.unmasked_files:
            person = os.path.basename(os.path.dirname(file))
            if person not in unmasked_by_person:
                unmasked_by_person[person] = []
            unmasked_by_person[person].append(file)
        
        self.masked_files = []
        self.unmasked_files = []
        
        for person, files in masked_by_person.items():
            selected = random.sample(files, min(len(files), self.max_samples_per_person))
            self.masked_files.extend(selected)
        
        for person, files in unmasked_by_person.items():
            selected = random.sample(files, min(len(files), self.max_samples_per_person))
            self.unmasked_files.extend(selected)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = self.files[idx]
        label = self.labels[idx]
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Не удалось загрузить изображение: {img_path}")
            img = cv2.resize(img, (112, 112))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            return img, label
            
        except Exception as e:
            logger.error(f"Ошибка при обработке изображения {img_path}: {str(e)}")
            random_idx = random.randint(0, len(self.files) - 1)
            return self.__getitem__(random_idx)

class FinetuneModel(nn.Module):
    def __init__(self, base_model):
        super(FinetuneModel, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        features_list = []
        for img in x:
            feat = self.base_model.get_feat(img)
            features_list.append(feat)
        
        features = np.vstack(features_list)
        features = torch.from_numpy(features).float()
        return self.classifier(features)

def prepare_model():
    logger.info("Загрузка предобученной модели buffalo_l")
    try:
        os.makedirs('./models', exist_ok=True)
        app = FaceAnalysis(name='buffalo_l', root='./models')
        app.prepare(ctx_id=0)  # GPU: ctx_id=0, CPU: ctx_id=-1
        finetune_model = FinetuneModel(app.models['recognition'])
        return finetune_model
        
    except Exception as e:
        logger.error(f"Ошибка при подготовке модели: {str(e)}")
        raise

def train_model(model, dataset, epochs=10, batch_size=32, learning_rate=0.001):
    logger.info("Начало дообучения модели...")
    train_files, val_files, train_labels, val_labels = train_test_split(
        dataset.files, dataset.labels, test_size=0.2, random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(dataset, range(len(train_files)))
    val_dataset = torch.utils.data.Subset(dataset, range(len(train_files), len(dataset)))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используется устройство: {device}")
    model = model.to(device)
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)  # только классификатор
    criterion = nn.CrossEntropyLoss()
    
    # обучение
    best_val_acc = 0.0
    for epoch in range(epochs):
        logger.info(f"Эпоха {epoch + 1}/{epochs}")
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc='Training'):
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # валидация
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation'):
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        logger.info(f"Эпоха {epoch + 1}: train_loss={train_loss/len(train_loader):.4f}, "
                   f"train_acc={train_acc:.2f}%, val_loss={val_loss/len(val_loader):.4f}, "
                   f"val_acc={val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_path)
            logger.info(f"Сохранена лучшая модель с точностью {val_acc:.2f}%")

    save_path = "finetuned_buffalo_l.pth"
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, save_path)
    logger.info(f"Финальная модель сохранена в {save_path}")

def main():
    dataset_path = "RMFD"
    try:
        if not os.path.exists(dataset_path):
            raise ValueError(f"Директория {dataset_path} не найдена")
        dataset = RMFDDataset(dataset_path)
        model = prepare_model()
        
        if model is None:
            raise ValueError("Не удалось подготовить модель")
        logger.info("Начало дообучения...")
        train_model(model, dataset)
        
    except Exception as e:
        logger.error(f"Произошла ошибка: {str(e)}")
        raise

if __name__ == "__main__":
    main() 