import json
import glob
import cv2
import numpy as np
from datetime import datetime
import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1  # Для FaceNet
import insightface  # Для InsightFace
from dlib_utils import init_dlib, get_dlib_embedding  # Импортируем наши функции
import torch
import os
import torch.nn as nn
from insightface.app import FaceAnalysis
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

with open('config.json') as f:
    config = json.load(f)


def init_models():
    # FaceNet
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
    mtcnn = MTCNN(keep_all=True, device='cpu')

    # dlib
    dlib_models = init_dlib()

    # InsightFace
    insight_model = insightface.app.FaceAnalysis()
    insight_model.prepare(ctx_id=0, det_size=(640, 640))

    return {
        'facenet': (facenet_model, mtcnn),
        'dlib': dlib_models,
        'insightface': insight_model
    }


def process_image(model_type, models, img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    if model_type == 'facenet':
        facenet_model, mtcnn = models['facenet']
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb_img)
        if boxes is None or len(boxes) == 0:
            return None
            
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(rgb_img.shape[1], x2)
        y2 = min(rgb_img.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        face = rgb_img[y1:y2, x1:x2]
        if face.size == 0:
            return None
            
        try:
            face = cv2.resize(face, (160, 160))
            face = torch.from_numpy(face).permute(2, 0, 1).float()
            face = face.unsqueeze(0)
            embedding = facenet_model(face).detach().numpy()
            return embedding[0]
        except Exception as e:
            print(f"Ошибка при обработке изображения {img_path}: {str(e)}")
            return None
    elif model_type == 'dlib':
        detector, sp, facerec = models['dlib']
        return get_dlib_embedding(img_path, detector, sp, facerec)
    elif model_type == 'insightface':
        insight_model = models['insightface']
        if isinstance(insight_model, FinetuneModel):
            result = process_image_with_finetuned_model(insight_model, img_path)
            if result is None:
                return None
            return insight_model.base_model.get_feat(img)
        else:
            faces = insight_model.get(img)
            if len(faces) == 0:
                return None
            return faces[0].embedding
    return None


def run_full_test(models):
    results = []
    print("Тестирование фронтальных изображений...")
    for img_path in glob.glob(config['test_sets']['frontal']):
        for model_type in ['facenet', 'dlib', 'insightface']:
            start_time = datetime.now()
            embedding = process_image(model_type, models, img_path)
            proc_time = (datetime.now() - start_time).total_seconds()

            results.append({
                'model': model_type,
                'test_type': 'frontal',
                'image': img_path,
                'time': proc_time,
                'detected': embedding is not None
            })

    print("Тестирование сложных случаев...")
    challenge_types = {
        'ac': 'accessories',
        'bad': 'bad_quality',
        'gl': 'glasses',
        'ms': 'mask'
    }
    
    for img_path in glob.glob(config['test_sets']['challenge']):
        for model_type in ['facenet', 'dlib', 'insightface']:
            start_time = datetime.now()
            embedding = process_image(model_type, models, img_path)
            proc_time = (datetime.now() - start_time).total_seconds()
            
            challenge_type = None
            for type_code, type_name in challenge_types.items():
                if type_code in img_path:
                    challenge_type = type_name
                    break
            
            results.append({
                'model': model_type,
                'test_type': f'challenge_{challenge_type}',
                'image': img_path,
                'time': proc_time,
                'detected': embedding is not None
            })

    print("Тестирование парных сравнений...")
    for pair_type in ['positive', 'negative']:
        pairs_dir = config['test_sets'][f'{pair_type}_pairs']
        for pair_num in range(1, 11):
            prefix = 'p' if pair_type == 'positive' else 'n'
            img1 = f"{pairs_dir}{prefix}{pair_num}_1.jpg"
            img2 = f"{pairs_dir}{prefix}{pair_num}_2.jpg"
            
            for model_type in ['facenet', 'dlib', 'insightface']:
                start_time = datetime.now()
                embedding1 = process_image(model_type, models, img1)
                embedding2 = process_image(model_type, models, img2)
                proc_time = (datetime.now() - start_time).total_seconds()
                
                distance = None
                if embedding1 is not None and embedding2 is not None:
                    distance = np.linalg.norm(embedding1 - embedding2)
                
                results.append({
                    'model': model_type,
                    'test_type': f'{pair_type}_pair',
                    'image1': img1,
                    'image2': img2,
                    'time': proc_time,
                    'detected': embedding1 is not None and embedding2 is not None,
                    'distance': distance
                })

    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(f'test_results_{timestamp}.csv', index=False)
    print(f"Результаты сохранены в test_results_{timestamp}.csv")
    return results

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

def load_finetuned_model():
    try:
        app = FaceAnalysis(name='buffalo_l', root='./models')
        app.prepare(ctx_id=0)
        
        model = FinetuneModel(app.models['recognition'])
        
        checkpoint = torch.load('finetuned_buffalo_l.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        
        return model
    except Exception as e:
        logger.error(f"Ошибка при загрузке дообученной модели: {str(e)}")
        return None

def process_image_with_finetuned_model(model, image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        with torch.no_grad():
            output = model(img)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        return {
            'prediction': prediction,  # 0 - без маски, 1 - с маской
            'confidence': confidence
        }
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения {image_path}: {str(e)}")
        return None

def run_tests():
    logger.info("Выберите сценарий тестирования:")
    logger.info("0 - Сравнение базовой модели insightface с другими моделями")
    logger.info("1 - Сравнение дообученной модели insightface с другими моделями")
    
    while True:
        try:
            choice = int(input("Введите 0 или 1: "))
            if choice in [0, 1]:
                break
            else:
                logger.error("Пожалуйста, введите 0 или 1")
        except ValueError:
            logger.error("Пожалуйста, введите число 0 или 1")
    
    if choice == 0:
        logger.info("Запуск тестов с базовой моделью insightface...")
        models = init_models()
    else:
        logger.info("Запуск тестов с дообученной моделью insightface...")
        base_models = init_models()
        finetuned_model = load_finetuned_model()
        if finetuned_model is None:
            logger.error("Не удалось загрузить дообученную модель")
            return
        models = {
            'facenet': base_models['facenet'],
            'dlib': base_models['dlib'],
            'insightface': finetuned_model
        }
    
    results = []
    
    logger.info("Тестирование фронтальных изображений...")
    for img_path in glob.glob('test_images/1front/*.jpg'):
        for model_type in ['facenet', 'dlib', 'insightface']:
            start_time = datetime.now()
            embedding = process_image(model_type, models, img_path)
            proc_time = (datetime.now() - start_time).total_seconds()
            
            results.append({
                'model': model_type,
                'test_type': 'frontal',
                'image': img_path,
                'time': proc_time,
                'detected': embedding is not None
            })
    
    logger.info("Тестирование сложных случаев...")
    challenge_types = {
        'ac': 'accessories',
        'bad': 'bad_quality',
        'gl': 'glasses',
        'ms': 'mask'
    }
    
    for img_path in glob.glob('test_images/2challenge/*.jpg'):
        for model_type in ['facenet', 'dlib', 'insightface']:
            start_time = datetime.now()
            embedding = process_image(model_type, models, img_path)
            proc_time = (datetime.now() - start_time).total_seconds()
            
            challenge_type = None
            for type_code, type_name in challenge_types.items():
                if type_code in img_path:
                    challenge_type = type_name
                    break
            
            results.append({
                'model': model_type,
                'test_type': f'challenge_{challenge_type}',
                'image': img_path,
                'time': proc_time,
                'detected': embedding is not None
            })
    
    logger.info("Тестирование парных сравнений...")
    for pair_type in ['positive', 'negative']:
        pairs_dir = f'test_images/3pairs/{"1positive" if pair_type == "positive" else "2negative"}/'
        for pair_num in range(1, 11):
            prefix = 'p' if pair_type == 'positive' else 'n'
            img1 = f"{pairs_dir}{prefix}{pair_num}_1.jpg"
            img2 = f"{pairs_dir}{prefix}{pair_num}_2.jpg"
            
            for model_type in ['facenet', 'dlib', 'insightface']:
                start_time = datetime.now()
                embedding1 = process_image(model_type, models, img1)
                embedding2 = process_image(model_type, models, img2)
                proc_time = (datetime.now() - start_time).total_seconds()
                
                distance = None
                if embedding1 is not None and embedding2 is not None:
                    distance = np.linalg.norm(embedding1 - embedding2)
                
                results.append({
                    'model': model_type,
                    'test_type': f'{pair_type}_pair',
                    'image1': img1,
                    'image2': img2,
                    'time': proc_time,
                    'detected': embedding1 is not None and embedding2 is not None,
                    'distance': distance
                })
    
    logger.info("\nРезультаты тестирования:")
    
    for model_type in ['facenet', 'dlib', 'insightface']:
        model_results = [r for r in results if r['model'] == model_type]
        
        test_types = {}
        for result in model_results:
            test_type = result['test_type']
            if test_type not in test_types:
                test_types[test_type] = {'total': 0, 'detected': 0, 'times': []}
            
            test_types[test_type]['total'] += 1
            if result['detected']:
                test_types[test_type]['detected'] += 1
            test_types[test_type]['times'].append(result['time'])
        
        logger.info(f"\n{model_type.upper()}:")
        for test_type, stats in test_types.items():
            accuracy = (stats['detected'] / stats['total'] * 100) if stats['total'] > 0 else 0
            avg_time = np.mean(stats['times']) if stats['times'] else 0
            
            logger.info(f"\n{test_type}:")
            logger.info(f"Всего тестов: {stats['total']}")
            logger.info(f"Успешно обнаружено: {stats['detected']}")
            logger.info(f"Точность: {accuracy:.2f}%")
            logger.info(f"Среднее время обработки: {avg_time:.4f} сек")
    
    save_results(results, choice)

def save_results(results, model_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_suffix = "finetuned" if model_type == 1 else "base"
    df_detailed = pd.DataFrame(results)
    detailed_csv_path = f'test_results_{model_suffix}_{timestamp}.csv'
    df_detailed.to_csv(detailed_csv_path, index=False)
    
    data_for_plots = []
    for model in ['facenet', 'dlib', 'insightface']:
        model_results = [r for r in results if r['model'] == model]
        test_types = {}
        
        for result in model_results:
            test_type = result['test_type']
            if test_type not in test_types:
                test_types[test_type] = {'total': 0, 'detected': 0, 'times': []}
            
            test_types[test_type]['total'] += 1
            if result['detected']:
                test_types[test_type]['detected'] += 1
            test_types[test_type]['times'].append(result['time'])
        
        for test_type, stats in test_types.items():
            accuracy = (stats['detected'] / stats['total'] * 100) if stats['total'] > 0 else 0
            avg_time = np.mean(stats['times']) if stats['times'] else 0
            
            data_for_plots.append({
                'Модель': model,
                'Тип теста': test_type,
                'Всего тестов': stats['total'],
                'Успешно обнаружено': stats['detected'],
                'Точность (%)': f"{accuracy:.2f}",
                'Среднее время (сек)': f"{avg_time:.4f}"
            })
    
    df_plots = pd.DataFrame(data_for_plots)
    plots_csv_path = f'for_png_{timestamp}.csv'
    df_plots.to_csv(plots_csv_path, index=False)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    accuracy_data = df_plots.pivot(index='Модель', columns='Тип теста', values='Точность (%)')
    accuracy_data = accuracy_data.astype(float)
    accuracy_data.plot(kind='bar', ax=plt.gca())
    plt.title('Точность по типам тестов')
    plt.ylabel('Точность (%)')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    time_data = df_plots.pivot(index='Модель', columns='Тип теста', values='Среднее время (сек)')
    time_data = time_data.astype(float)
    time_data.plot(kind='bar', ax=plt.gca())
    plt.title('Среднее время обработки')
    plt.ylabel('Время (сек)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'test_results_{model_suffix}_{timestamp}.png')
    
    logger.info(f"\nРезультаты сохранены:")
    logger.info(f"Детальный CSV: {detailed_csv_path}")
    logger.info(f"CSV для графиков: {plots_csv_path}")
    logger.info(f"Графики: test_results_{model_suffix}_{timestamp}.png")

if __name__ == "__main__":
    run_tests()