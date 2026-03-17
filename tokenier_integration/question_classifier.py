"""
Question Type Classifier using BPE Tokenizer + ML Classifier
Классификация типов вопросов (фактический/процедурный/юридическая_интерпретация/сравнение/да_нет)
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


class QuestionClassifier:
    """
    Классификатор типов вопросов
    
    Типы вопросов:
    - factual: Фактические вопросы (Что? Кто? Когда? Где?)
    - procedural: Процедурные вопросы (Как? Каким образом?)
    - legal_interpretation: Юридическая интерпретация (Означает ли? Применимо ли?)
    - comparison: Сравнительные вопросы (В чем разница? Что лучше?)
    - yes_no: Вопросы да/нет (Может ли? Должен ли?)
    """
    
    QUESTION_TYPES = ['factual', 'procedural', 'legal_interpretation', 'comparison', 'yes_no']
    
    # Параметры поиска для каждого типа вопроса
    SEARCH_PARAMS = {
        'factual': {
            'top_k': 5,
            'rerank': True,
            'expand_query': False,
            'description': 'Точный поиск фактов'
        },
        'procedural': {
            'top_k': 10,
            'rerank': True,
            'expand_query': True,
            'description': 'Поиск процедур и инструкций'
        },
        'legal_interpretation': {
            'top_k': 7,
            'rerank': True,
            'expand_query': True,
            'description': 'Поиск юридических толкований'
        },
        'comparison': {
            'top_k': 8,
            'rerank': True,
            'expand_query': False,
            'description': 'Поиск для сравнения'
        },
        'yes_no': {
            'top_k': 3,
            'rerank': False,
            'expand_query': False,
            'description': 'Быстрый поиск для да/нет'
        }
    }
    
    def __init__(
        self,
        tokenizer_path: str = "models/tokenier/tokenizer.pkl",
        model_path: Optional[str] = None,
        classifier_type: str = "xgboost",
        embedding_dim: int = 128
    ):
        """
        Args:
            tokenizer_path: Путь к обученному BPE токенизатору
            model_path: Путь к обученному классификатору
            classifier_type: Тип классификатора
            embedding_dim: Размерность эмбеддингов
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
    
    def _extract_features(self, question: str) -> np.ndarray:
        """
        Извлечение признаков из вопроса
        
        Признаки:
        1. Token features (токены вопроса)
        2. Question word features (вопросительные слова)
        3. Syntactic features (синтаксические признаки)
        4. Length features (длина вопроса)
        
        Args:
            question: Текст вопроса
            
        Returns:
            Вектор признаков
        """
        # Токенизация
        token_ids = self.tokenizer.encode(question)
        
        if len(token_ids) == 0:
            return np.zeros(self.embedding_dim)
        
        # 1. Token frequency features
        vocab_size = self.tokenizer.get_vocab_size()
        token_freq = np.zeros(min(vocab_size, 500))
        for tid in token_ids:
            if tid < len(token_freq):
                token_freq[tid] += 1
        token_freq = token_freq / (len(token_ids) + 1e-10)
        
        # 2. Question word features (вопросительные слова)
        question_lower = question.lower()
        question_words = {
            'what': ['что', 'какой', 'какая', 'какое', 'какие'],
            'who': ['кто', 'кого', 'кому', 'кем'],
            'when': ['когда'],
            'where': ['где', 'куда', 'откуда'],
            'why': ['почему', 'зачем'],
            'how': ['как', 'каким образом'],
            'which': ['который', 'которая', 'которое', 'которые'],
            'can': ['может', 'могут', 'можно'],
            'should': ['должен', 'должна', 'должно', 'должны'],
            'is': ['является', 'есть', 'ли']
        }
        
        qword_features = []
        for qtype, words in question_words.items():
            has_word = any(word in question_lower for word in words)
            qword_features.append(1.0 if has_word else 0.0)
        qword_features = np.array(qword_features)
        
        # 3. Syntactic features
        has_question_mark = 1.0 if '?' in question else 0.0
        starts_with_qword = 1.0 if any(
            question_lower.startswith(word)
            for words in question_words.values()
            for word in words
        ) else 0.0
        
        # Наличие сравнительных слов
        comparison_words = ['разница', 'отличие', 'сравнить', 'лучше', 'хуже', 'больше', 'меньше']
        has_comparison = 1.0 if any(word in question_lower for word in comparison_words) else 0.0
        
        # Наличие юридических терминов
        legal_words = ['закон', 'право', 'статья', 'кодекс', 'норма', 'применим', 'означает']
        has_legal = 1.0 if any(word in question_lower for word in legal_words) else 0.0
        
        syntactic_features = np.array([
            has_question_mark,
            starts_with_qword,
            has_comparison,
            has_legal
        ])
        
        # 4. Length features
        length_features = np.array([
            len(token_ids),
            len(question.split()),
            len(question)
        ])
        
        # 5. Bigram features
        bigrams = {}
        for i in range(len(token_ids) - 1):
            bigram = (token_ids[i], token_ids[i + 1])
            bigrams[bigram] = bigrams.get(bigram, 0) + 1
        
        top_bigrams = sorted(bigrams.items(), key=lambda x: x[1], reverse=True)[:30]
        bigram_features = np.array([count for _, count in top_bigrams] + [0] * (30 - len(top_bigrams)))
        bigram_features = bigram_features / (len(token_ids) + 1e-10)
        
        # Объединение признаков
        target_size = self.embedding_dim - len(qword_features) - len(syntactic_features) - len(length_features) - len(bigram_features)
        if target_size > 0:
            token_freq_trimmed = token_freq[:target_size]
            if len(token_freq_trimmed) < target_size:
                token_freq_trimmed = np.pad(token_freq_trimmed, (0, target_size - len(token_freq_trimmed)))
        else:
            token_freq_trimmed = np.array([])
        
        features = np.concatenate([
            token_freq_trimmed,
            qword_features,
            syntactic_features,
            length_features,
            bigram_features
        ])
        
        # Обрезаем или дополняем до embedding_dim
        if len(features) > self.embedding_dim:
            features = features[:self.embedding_dim]
        elif len(features) < self.embedding_dim:
            features = np.pad(features, (0, self.embedding_dim - len(features)))
        
        return features
    
    def train(
        self,
        questions: List[str],
        labels: List[str],
        test_size: float = 0.2,
        n_estimators: int = 150,
        max_depth: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Обучение классификатора
        
        Args:
            questions: Список вопросов
            labels: Список меток (типов вопросов)
            test_size: Доля тестовой выборки
            n_estimators: Количество деревьев
            max_depth: Максимальная глубина деревьев
            verbose: Вывод информации
            
        Returns:
            Словарь с метриками качества
        """
        if len(questions) != len(labels):
            raise ValueError("questions and labels must have the same length")
        
        if verbose:
            print(f"Extracting features from {len(questions)} questions...")
        
        # Извлечение признаков
        X = np.array([self._extract_features(q) for q in questions])
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
        
        # Нормализация
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Кодирование меток
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        if verbose:
            print(f"Training {self.classifier_type} classifier...")
        
        # Обучение
        if self.classifier_type == "xgboost" and HAS_XGB:
            self.classifier = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth or 5,
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
        
        # Оценка
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
    
    def predict(self, question: str) -> str:
        """Предсказание типа вопроса"""
        if self.classifier is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")
        
        features = self._extract_features(question).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        pred_encoded = self.classifier.predict(features_scaled)[0]
        pred_label = self.label_encoder.inverse_transform([pred_encoded])[0]
        
        return pred_label
    
    def predict_with_params(self, question: str) -> Tuple[str, Dict]:
        """
        Предсказание типа вопроса с рекомендуемыми параметрами поиска
        
        Returns:
            (question_type, search_params)
        """
        question_type = self.predict(question)
        search_params = self.SEARCH_PARAMS.get(question_type, self.SEARCH_PARAMS['factual'])
        
        return question_type, search_params
    
    def predict_proba(self, question: str) -> Dict[str, float]:
        """Предсказание вероятностей для каждого типа вопроса"""
        if self.classifier is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")
        
        features = self._extract_features(question).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        probas = self.classifier.predict_proba(features_scaled)[0]
        
        result = {}
        for i, label in enumerate(self.label_encoder.classes_):
            result[label] = float(probas[i])
        
        return result
    
    def save_model(self, path: str):
        """Сохранение модели"""
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
        """Загрузка модели"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        
        bundle = joblib.load(path)
        
        self.classifier = bundle['classifier']
        self.scaler = bundle['scaler']
        self.label_encoder = bundle['label_encoder']
        self.classifier_type = bundle.get('classifier_type', 'random_forest')
        self.embedding_dim = bundle.get('embedding_dim', 128)
        
        print(f"Model loaded from {path}")
