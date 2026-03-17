"""
Document Type Classifier using BPE Tokenizer + ML Classifier
Классификация типов юридических документов (закон/дело/регламент/указ/поправка)
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from .bpe_tokenizer import BPETokenizer


class DocumentClassifier:
    """
    Классификатор типов юридических документов
    
    Типы документов:
    - law: Законы
    - case: Судебные дела
    - regulation: Регламенты
    - decree: Указы
    - amendment: Поправки
    """
    
    DOCUMENT_TYPES = ['law', 'case', 'regulation', 'decree', 'amendment']
    
    def __init__(
        self,
        tokenizer_path: str = "models/tokenier/tokenizer.pkl",
        model_path: Optional[str] = None,
        classifier_type: str = "xgboost",  # "random_forest" or "xgboost"
        embedding_dim: int = 256
    ):
        """
        Args:
            tokenizer_path: Путь к обученному BPE токенизатору
            model_path: Путь к обученному классификатору (если None, нужно обучить)
            classifier_type: Тип классификатора ("random_forest" или "xgboost")
            embedding_dim: Размерность эмбеддингов для токенов
        """
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
        self.classifier_type = classifier_type
        self.embedding_dim = embedding_dim
        
        # Загрузка токенизатора
        self.tokenizer = self._load_tokenizer()
        
        # Инициализация классификатора
        self.classifier = None
        self.scaler = None
        self.label_encoder = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _load_tokenizer(self) -> BPETokenizer:
        """Загрузка обученного BPE токенизатора"""
        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {self.tokenizer_path}")
        
        tokenizer = BPETokenizer()
        tokenizer.load(self.tokenizer_path)
        return tokenizer
    
    def _extract_features(self, text: str) -> np.ndarray:
        """
        Извлечение признаков из текста
        
        Признаки:
        1. Token frequency features (частоты токенов)
        2. Token n-gram features (биграммы, триграммы)
        3. Statistical features (длина, уникальность токенов)
        4. Domain-specific features (юридические термины)
        
        Args:
            text: Входной текст документа
            
        Returns:
            Вектор признаков
        """
        # Токенизация
        token_ids = self.tokenizer.encode(text)
        
        if len(token_ids) == 0:
            return np.zeros(self.embedding_dim)
        
        # 1. Token frequency features (гистограмма токенов)
        vocab_size = self.tokenizer.get_vocab_size()
        token_freq = np.zeros(min(vocab_size, 1000))  # Ограничиваем размер
        for tid in token_ids:
            if tid < len(token_freq):
                token_freq[tid] += 1
        token_freq = token_freq / (len(token_ids) + 1e-10)  # Нормализация
        
        # 2. Statistical features
        stat_features = np.array([
            len(token_ids),  # Длина в токенах
            len(set(token_ids)),  # Уникальные токены
            len(set(token_ids)) / (len(token_ids) + 1e-10),  # Уникальность
            np.mean(token_ids) if token_ids else 0,  # Средний ID токена
            np.std(token_ids) if len(token_ids) > 1 else 0,  # Стд ID токена
        ])
        
        # 3. Bigram features (топ-50 биграмм)
        bigrams = {}
        for i in range(len(token_ids) - 1):
            bigram = (token_ids[i], token_ids[i + 1])
            bigrams[bigram] = bigrams.get(bigram, 0) + 1
        
        # Топ-50 биграмм
        top_bigrams = sorted(bigrams.items(), key=lambda x: x[1], reverse=True)[:50]
        bigram_features = np.array([count for _, count in top_bigrams] + [0] * (50 - len(top_bigrams)))
        bigram_features = bigram_features / (len(token_ids) + 1e-10)
        
        # 4. Domain-specific features (юридические ключевые слова)
        legal_keywords = {
            'law': ['закон', 'статья', 'кодекс', 'право', 'норма'],
            'case': ['дело', 'суд', 'решение', 'истец', 'ответчик', 'приговор'],
            'regulation': ['регламент', 'порядок', 'процедура', 'правило'],
            'decree': ['указ', 'постановление', 'распоряжение'],
            'amendment': ['поправка', 'изменение', 'дополнение', 'редакция']
        }
        
        text_lower = text.lower()
        keyword_features = []
        for doc_type, keywords in legal_keywords.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            keyword_features.append(count)
        keyword_features = np.array(keyword_features)
        
        # Объединение всех признаков
        # Обрезаем token_freq до нужного размера
        target_size = self.embedding_dim - len(stat_features) - len(bigram_features) - len(keyword_features)
        if target_size > 0:
            token_freq_trimmed = token_freq[:target_size]
            if len(token_freq_trimmed) < target_size:
                token_freq_trimmed = np.pad(token_freq_trimmed, (0, target_size - len(token_freq_trimmed)))
        else:
            token_freq_trimmed = np.array([])
        
        features = np.concatenate([
            token_freq_trimmed,
            stat_features,
            bigram_features,
            keyword_features
        ])
        
        # Обрезаем или дополняем до embedding_dim
        if len(features) > self.embedding_dim:
            features = features[:self.embedding_dim]
        elif len(features) < self.embedding_dim:
            features = np.pad(features, (0, self.embedding_dim - len(features)))
        
        return features
    
    def train(
        self,
        texts: List[str],
        labels: List[str],
        test_size: float = 0.2,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Обучение классификатора
        
        Args:
            texts: Список текстов документов
            labels: Список меток (типов документов)
            test_size: Доля тестовой выборки
            n_estimators: Количество деревьев (для RF/XGBoost)
            max_depth: Максимальная глубина деревьев
            verbose: Вывод информации о процессе
            
        Returns:
            Словарь с метриками качества
        """
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        
        if verbose:
            print(f"Extracting features from {len(texts)} documents...")
        
        # Извлечение признаков
        X = np.array([self._extract_features(text) for text in texts])
        y = np.array(labels)
        
        # Разбиение на train/test
        n_test = int(len(X) * test_size)
        indices = np.random.permutation(len(X))
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        if verbose:
            print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Нормализация признаков
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Кодирование меток
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        if verbose:
            print(f"Training {self.classifier_type} classifier...")
        
        # Обучение классификатора
        if self.classifier_type == "xgboost" and HAS_XGB:
            self.classifier = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth or 6,
                learning_rate=0.1,
                eval_metric='mlogloss',
                random_state=42
            )
        else:
            self.classifier = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
        
        self.classifier.fit(X_train_scaled, y_train_encoded)
        
        # Оценка качества
        y_pred = self.classifier.predict(X_test_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y_test_encoded, y_pred),
            'precision': precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        }
        
        if verbose:
            print(f"\nClassification Results:")
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1 Score:  {metrics['f1']:.4f}")
            print(f"\nDetailed Report:")
            present_labels = np.unique(np.concatenate([y_test_encoded, y_pred]))
            print(classification_report(
                y_test_encoded, y_pred,
                labels=present_labels,
                target_names=[self.label_encoder.classes_[i] for i in present_labels],
                zero_division=0
            ))
        
        return metrics
    
    def predict(self, text: str) -> str:
        """
        Предсказание типа документа
        
        Args:
            text: Текст документа
            
        Returns:
            Тип документа (law/case/regulation/decree/amendment)
        """
        if self.classifier is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")
        
        # Извлечение признаков
        features = self._extract_features(text).reshape(1, -1)
        
        # Нормализация
        features_scaled = self.scaler.transform(features)
        
        # Предсказание
        pred_encoded = self.classifier.predict(features_scaled)[0]
        pred_label = self.label_encoder.inverse_transform([pred_encoded])[0]
        
        return pred_label
    
    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Предсказание вероятностей для каждого типа документа
        
        Args:
            text: Текст документа
            
        Returns:
            Словарь {тип_документа: вероятность}
        """
        if self.classifier is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")
        
        # Извлечение признаков
        features = self._extract_features(text).reshape(1, -1)
        
        # Нормализация
        features_scaled = self.scaler.transform(features)
        
        # Предсказание вероятностей
        probas = self.classifier.predict_proba(features_scaled)[0]
        
        # Создание словаря
        result = {}
        for i, label in enumerate(self.label_encoder.classes_):
            result[label] = float(probas[i])
        
        return result
    
    def save_model(self, path: str):
        """Сохранение обученной модели"""
        if self.classifier is None:
            raise ValueError("No model to save. Train the model first.")
        
        bundle = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'classifier_type': self.classifier_type,
            'embedding_dim': self.embedding_dim,
            'tokenizer_path': self.tokenizer_path
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(bundle, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Загрузка обученной модели"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        
        bundle = joblib.load(path)
        
        self.classifier = bundle['classifier']
        self.scaler = bundle['scaler']
        self.label_encoder = bundle['label_encoder']
        self.classifier_type = bundle.get('classifier_type', 'random_forest')
        self.embedding_dim = bundle.get('embedding_dim', 256)
        
        print(f"Model loaded from {path}")
