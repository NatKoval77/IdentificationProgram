import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import pickle
import time
import sys
import io
import contextlib

def download_models():
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            face_analyzer = FaceAnalysis(
                name='buffalo_l',
                root='.',
                providers=['CPUExecutionProvider']
            )
            face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        print("Модель загружена")
        return face_analyzer
    except Exception as e:
        print(f"Ошибка при загрузке моделей: {str(e)}")
        raise

class FaceRecognition:
    def __init__(self):
        try:
            # основной анализатор
            self.face_analyzer = download_models()
            with contextlib.redirect_stdout(io.StringIO()):
                self.face_analyzer.prepare(
                    ctx_id=0,
                    det_size=(640, 640),
                    det_thresh=0.5
                )
            
            # альтернативный анализатор
            self.alt_face_analyzer = FaceAnalysis(
                name='buffalo_l',
                root='.',
                providers=['CPUExecutionProvider']
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.alt_face_analyzer.prepare(
                    ctx_id=0,
                    det_size=(320, 320),
                    det_thresh=0.3
                )
            
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            
            self.face_database = {}
            self.face_embeddings = []
            self.face_names = []
            
            # директория для бд
            if not os.path.exists('faces_db'):
                os.makedirs('faces_db')
            self.load_database()

            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            with contextlib.redirect_stdout(io.StringIO()):
                for _ in range(3):
                    self.face_analyzer.get(dummy_image)
                    self.alt_face_analyzer.get(dummy_image)
            
        except Exception as e:
            print(f"Ошибка при инициализации системы идентификации лиц: {str(e)}")
            raise
        
    def load_database(self):
        try:
            db_file = os.path.join('faces_db', 'face_database.pkl')
            if os.path.exists(db_file):
                with open(db_file, 'rb') as f:
                    data = pickle.load(f)
                    self.face_database = data['face_database']
                    
                    # пересоздание списков
                    self.face_embeddings = []
                    self.face_names = []
                    for name, embeddings in self.face_database.items():
                        self.face_embeddings.extend(embeddings)
                        self.face_names.extend([name] * len(embeddings))
                    print(f"Загружено {len(self.face_embeddings)} эмбеддингов")
            else:
                self.face_database = {}
                self.face_embeddings = []
                self.face_names = []
                self.save_database()
                print("Создана новая база данных")
        except Exception as e:
            print(f"Ошибка при загрузке базы данных: {str(e)}")
            self.face_database = {}
            self.face_embeddings = []
            self.face_names = []
                
    def save_database(self):
        try:
            db_file = os.path.join('faces_db', 'face_database.pkl')
            with open(db_file, 'wb') as f:
                pickle.dump({
                    'face_database': self.face_database,
                    'face_embeddings': self.face_embeddings,
                    'face_names': self.face_names
                }, f)
            print(f"Сохранено {len(self.face_embeddings)} эмбеддингов")
        except Exception as e:
            print(f"Ошибка при сохранении базы данных: {str(e)}")
            
    def extract_face_embedding(self, face_img):
        try:
            if face_img is None or face_img.size == 0:
                print("Ошибка: пустое изображение")
                return None
                
            if face_img.shape[0] < 20 or face_img.shape[1] < 20:
                print("Ошибка: изображение слишком маленькое")
                return None
                
            print(f"Размер обработанного изображения: {face_img.shape}")
            faces = self.face_analyzer.get(face_img)
            print(f"Найдено основным анализатором: {len(faces)}")
            if not faces:
                faces = self.alt_face_analyzer.get(face_img)
                print(f"Найдено альтернативным анализатором: {len(faces)}")
            
            if len(faces) > 0:
                face = faces[0]
                print(f"Размер области лица: {face.bbox}")
                
                if hasattr(face, 'embedding') and face.embedding is not None:
                    embedding = face.embedding
                    if len(embedding) > 0:
                        print(f"Успешно извлечен эмбеддинг размером {len(embedding)}")
                        return embedding
                    else:
                        print("Эмбеддинг пустой")
                else:
                    print("У лица нет атрибута embedding")
            else:
                print("Лица не найдены на изображении")
            return None
            
        except Exception as e:
            print(f"Ошибка при извлечении эмбеддинга: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
    def apply_clahe(self, image):
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            cl = self.clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        except Exception as e:
            print(f"Ошибка CLAHE: {str(e)}")
            return image

    def try_recognize_with_analyzer(self, image, analyzer, is_alternative=False):
        try:
            print(f"Попытка распознавания с {'альтернативным' if is_alternative else 'основным'} детектором")
            faces = analyzer.get(image)
            print(f"Найдено лиц: {len(faces)}")
            if len(faces) == 0:
                print("Лица не найдены на изображении")
                return None
            results = []

            for i, face in enumerate(faces, 1):
                print(f"\nОбработка лица {i}:")
                print(f"Размер области лица: {face.bbox}")
                if hasattr(face, 'embedding') and face.embedding is not None:
                    embedding = face.embedding
                    if len(embedding) == 0:
                        print("Ошибка: эмбеддинг пустой")
                        continue
                    print(f"Размер эмбеддинга: {len(embedding)}")
                else:
                    print("Ошибка: у лица нет атрибута embedding")
                    continue
                
                # норм
                embedding = embedding / np.linalg.norm(embedding)
                
                # поиск ближайшего
                min_dist = float('inf')
                recognized_name = "Unknown"
                recognition_threshold = 0.8
                print("\nСравнение с базой данных:")
                print(f"Порог распознавания: {recognition_threshold}")
                for name, stored_embeddings in self.face_database.items():
                    print(f"\nСравнение с {name}:")
                    for j, stored_embedding in enumerate(stored_embeddings, 1):
                        stored_embedding = stored_embedding / np.linalg.norm(stored_embedding)
                        dist = 1 - np.dot(embedding, stored_embedding)
                        print(f"- Эмбеддинг {j}: {dist:.4f}")
                        
                        if dist < min_dist:
                            min_dist = dist
                            recognized_name = name
                
                print(f"\nРезультат сравнения:")
                print(f"Минимальное расстояние: {min_dist:.4f}")
                
                if min_dist > recognition_threshold:
                    recognized_name = "Unknown"
                    print("Расстояние превышает порог распознавания")
                else:
                    print(f"Распознано как: {recognized_name}")
                
                results.append((recognized_name, ))

            if results:
                return results[0][0]
            return None
            
        except Exception as e:
            print(f"Ошибка при распознавании: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def try_detect_with_analyzer(self, image, analyzer, is_alternative=False):
        """Попытка обнаружения лиц с помощью указанного детектора"""
        print(f"\nПопытка обнаружения лиц с {'альтернативным' if is_alternative else 'основным'} детектором (CLAHE: Да):")

        enhanced_image = self.apply_clahe(image)
        faces = analyzer.get(enhanced_image)
        if len(faces) > 0:
            print(f"Обнаружено лиц: {len(faces)}")
            return faces
        else:
            print("Лица не обнаружены")
            return None

    def recognize_face(self, image):
        results = []
        try:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print("Начало распознавания лиц")
            # основной
            faces = self.try_detect_with_analyzer(image, self.face_analyzer)
            # альтернативный
            if faces is None or not faces:
                print("\nПробуем обнаружить с альтернативным")
                faces = self.try_detect_with_analyzer(image, self.alt_face_analyzer, is_alternative=True)
            if faces is None or not faces:
                print("Лица не обнаружены ни одним из детекторов.")
                return []
            
            print(f"Обнаружено {len(faces)} лиц для распознавания.")
            
            for i, face in enumerate(faces):
                print(f"\nОбработка лица {i+1} из {len(faces)}:")
                if not hasattr(face, 'embedding') or face.embedding is None:
                    print(f"Лицо {i+1}: Не удалось извлечь эмбеддинг.")
                    continue
                embedding = face.embedding
                if len(embedding) == 0:
                    print(f"Лицо {i+1}: Эмбеддинг пустой.")
                    continue
                
                # норм
                embedding = embedding / np.linalg.norm(embedding)
                min_dist = float('inf')
                recognized_name = "Unknown"
                recognition_threshold = 0.8
                
                if self.face_database:
                    print("\nСравнение с базой данных:")
                    print(f"Порог распознавания: {recognition_threshold}")
                    for name, stored_embeddings in self.face_database.items():
                        print(f"\nСравнение с {name}:")
                        for j, stored_embedding in enumerate(stored_embeddings, 1):
                            stored_embedding = stored_embedding / np.linalg.norm(stored_embedding)
                            dist = 1 - np.dot(embedding, stored_embedding)
                            print(f"- Эмбеддинг {j}: {dist:.4f}")
                            
                            if dist < min_dist:
                                min_dist = dist
                                recognized_name = name
                
                    print(f"\nРезультат сравнения:")
                    print(f"Минимальное расстояние: {min_dist:.4f}")
                    if min_dist > recognition_threshold:
                        recognized_name = "Unknown"
                        print("Расстояние превышает порог распознавания")
                    else:
                        print(f"Распознано как: {recognized_name}")
                else:
                    recognized_name = "Unknown"
                    print("База данных лиц пуста. Невозможно выполнить сравнение.")
                results.append((recognized_name, face.bbox))

            print(f"Итоговый результат: Распознано {len(results)} лиц.")
            return results
            
        except Exception as e:
            print(f"Ошибка при распознавании лиц: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
        
    def add_face(self, face_img, name):
        try:
            embedding = self.extract_face_embedding(face_img)
            if embedding is None:
                print("Не удалось извлечь эмбеддинг")
                return False
                
            # норм
            embedding = embedding / np.linalg.norm(embedding)
            if name not in self.face_database:
                self.face_database[name] = []
            
            # проверка на похожесть
            is_duplicate = False
            for stored_embedding in self.face_database[name]:
                stored_embedding = stored_embedding / np.linalg.norm(stored_embedding)
                similarity = np.dot(embedding, stored_embedding)
                if similarity > 0.95:  # если слишком похожи
                    is_duplicate = True
                    print(f"Эмбеддинг похож на существующий")
                    break
            
            if not is_duplicate:
                self.face_database[name].append(embedding)
                print(f"Добавлен новый эмбеддинг для {name}")
                print(f"Всего эмбеддингов для {name}: {len(self.face_database[name])}")
                self.face_embeddings.append(embedding)
                self.face_names.append(name)
                self.save_database()
                return True
            else:
                print("Эмбеддинг не добавлен - слишком похож на существующий")
                return False
                
        except Exception as e:
            print(f"Ошибка при добавлении лица: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
    def collect_faces(self, frame, name, max_frames=30):
        try:
            faces = self.face_analyzer.get(frame)
            if len(faces) > 0:
                return self.add_face(frame, name)
            return False
        except Exception as e:
            print(f"Ошибка при сборе лиц: {str(e)}")
            return False

    def process_video_frames(self, frames, name):
        if not frames or len(frames) < 10:
            print(f"Недостаточно кадров для обработки: {len(frames)}")
            return False
            
        embeddings_list = []
        
        print(f"\nНачало обработки {len(frames)} кадров")
        
        for i, frame in enumerate(frames, 1):
            try:
                if frame.shape[0] < 20 or frame.shape[1] < 20:
                    print(f"Кадр {i} слишком маленький")
                    continue
                    
                faces = self.face_analyzer.get(frame)
                if len(faces) > 0:
                    face = faces[0]
                    if hasattr(face, 'embedding') and face.embedding is not None:
                        embedding = face.embedding
                        if len(embedding) > 0:
                            # норм
                            embedding = embedding / np.linalg.norm(embedding)
                            embeddings_list.append(embedding)
                            print(f"Обработан кадр {i}/{len(frames)}")
            except Exception as e:
                print(f"Ошибка при обработке кадра {i}: {str(e)}")
                continue
        
        if len(embeddings_list) < 10:
            print(f"Недостаточно эмбеддингов для усреднения: {len(embeddings_list)}")
            return False
        
        print(f"\nСоздание усредненных эмбеддингов")
        grouped_embeddings = []
        for i in range(0, len(embeddings_list), 10):
            group = embeddings_list[i:i+10]
            if len(group) == 10:
                avg_embedding = np.mean(group, axis=0)
                avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
                grouped_embeddings.append(avg_embedding)
                print(f"Создан усредненный эмбеддинг {len(grouped_embeddings)}")

        remaining = len(embeddings_list) % 10
        if remaining > 0:
            remaining_embeddings = embeddings_list[-remaining:]
            avg_remaining = np.mean(remaining_embeddings, axis=0)
            avg_remaining = avg_remaining / np.linalg.norm(avg_remaining)
            grouped_embeddings.append(avg_remaining)
            print(f"Создан усредненный эмбеддинг из оставшихся {remaining} эмбеддингов")

        if name not in self.face_database:
            self.face_database[name] = []
        self.face_database[name].extend(grouped_embeddings)
        self.face_embeddings.extend(grouped_embeddings)
        self.face_names.extend([name] * len(grouped_embeddings))
        self.save_database()
        print(f"\nСоздано {len(grouped_embeddings)} усредненных эмбеддингов для {name}")
        return True

    def get_single_face_for_addition(self, image):
        faces = self.try_detect_with_analyzer(image, self.face_analyzer)
        if faces is None or not faces:
            faces = self.try_detect_with_analyzer(image, self.alt_face_analyzer, is_alternative=True)
        if faces and len(faces) == 1:
            print("Найдено одно лицо для добавления.")
            return faces[0]
        elif faces and len(faces) > 1:
            print(f"Найдено несколько лиц ({len(faces)}). Требуется одно лицо для добавления.")
            return None
        else:
            print("Лицо не найдено ни одним из детекторов.")
            return None