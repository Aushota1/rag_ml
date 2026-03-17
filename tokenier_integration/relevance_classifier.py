"""
Relevance Classifier using BPE Tokenizer + ML Classifier
Бинарная классификация релевантности (вопрос, чанк) -> релевантен/нерелевантен
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from .bpe_tokenizer import BPETokenizer


class RelevanceClassifier:
    """
    Бинарный классификатор релевантности пары (вопрос, чанк)
    
    Используется как дополнительный фильтр после reranker для уменьшения false positives
    """
    
    def __init__(
        self,
        tokenizer_path: str = "models/tokenier/tokenizer.pkl",
        model_path: Optional[str] = None,
        classifier_type: str = "xgboost",
        embedding_dim: int = 256
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
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _load_tokenizer(self) -> BPETokenizer:
        """Загрузка обученного BPE токенизатора"""
        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {self.tokenizer_path}")
        
        tokenizer = BPETokenizer()
        tokenizer.load(self.tokenizer_path)
        return tokenizer
    
    def _extract_features(self, question: str, chunk: str) -> np.ndarray:
        """
        Извлечение признаков из пары (вопрос, чанк)
        
        Признаки:
        1. Token overlap features (пересечение токенов)
        2. Semantic similarity features (семантическая близость)
        3. Length features (длина вопроса и чанка)
        4. Statistical features (статистические признаки)
        
        Args:
            question: Текст вопроса
            chunk: Текст чанка
            
        Returns:
            Вектор признаков
        """
        # Токенизация
        question_tokens = self.tokenizer.encode(question)
        chunk_tokens = self.tokenizer.encode(chunk)
        
        if len(question_tokens) == 0 or len(chunk_tokens) == 0:
            return np.zeros(self.embedding_dim)
        
        # 1. Token overlap features
        question_set = set(question_tokens)
        chunk_set = set(chunk_tokens)
        
        intersection = question_set & chunk_set
        union = question_set | chunk_set
        
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        overlap_ratio_q = len(intersection) / len(question_set) if question_set else 0.0
        overlap_ratio_c = len(intersection) / len(chunk_set) if chunk_set else 0.0
        
        overlap_features = np.array([
            jaccard_similarity,
            overlap_ratio_q,
            overlap_ratio_c,
            len(intersection),
            len(union)
        ])
        
        # 2. Token frequency features (для вопроса и чанка)
        vocab_size = self.tokenizer.get_vocab_size()
        max_vocab = min(vocab_size, 300)
        
        question_freq = np.zeros(max_vocab)
        for tid in question_tokens:
            if tid < max_vocab:
                question_freq[tid] += 1
        question_freq = question_freq / (len(question_tokens) + 1e-10)
        
        chunk_freq = np.zeros(max_vocab)
        for tid in chunk_tokens:
            if tid < max_vocab:
                chunk_freq[tid] += 1
        chunk_freq = chunk_freq / (len(chunk_tokens) + 1e-10)
        
        # Косинусная близость частот токенов
        freq_similarity = np.dot(question_freq, chunk_freq) / (
            np.linalg.norm(question_freq) * np.linalg.norm(chunk_freq) + 1e-10
        )
        
        # 3. Length features
        length_features = np.array([
            len(question_tokens),
            len(chunk_tokens),
            len(chunk_tokens) / (len(question_tokens) + 1e-10),
            abs(len(chunk_tokens) - len(question_tokens))
        ])
        
        # 4. Statistical features
        # Позиция первого совпадения
        first_match_pos = -1
        for i, token in enumerate(chunk_tokens):
            if token in question_set:
                first_match_pos = i / len(chunk_tokens)
                break
        
        # Плотность совпадений
        match_positions = [i for i, token in enumerate(chunk_tokens) if token in question_set]
        match_density = len(match_positions) / len(chunk_tokens) if chunk_tokens else 0.0
        
        # Средняя позиция совпадений
        avg_match_pos = np.mean(match_positions) / len(chunk_tokens) if match_positions else 0.5
        
        stat_features = np.array([
            first_match_pos,
            match_density,
            avg_match_pos,
            freq_similarity
        ])
        
        # 5. N-gram overlap features
        # Биграммы вопроса
        question_bigrams = set()
        for i in range(len(question_tokens) - 1):
            question_bigrams.add((question_tokens[i], question_tokens[i + 1]))
        
        # Биграммы чанка
        chunk_bigrams = set()
        for i in range(len(chunk_tokens) - 1):
            chunk_bigrams.add((chunk_tokens[i], chunk_tokens[i + 1]))
        
        bigram_intersection = question_bigrams & chunk_bigrams
        bigram_union = question_bigrams | chunk_bigrams
        
        bigram_jaccard = len(bigram_intersection) / len(bigram_union) if bigram_union else 0.0
        
        ngram_features = np.array([
            bigram_jaccard,
            len(bigram_intersection),
            len(question_bigrams),
            len(chunk_bigrams)
        ])
        
        # 6. Keyword matching features (юридические термины)
        legal_keywords = [
            'закон', 'статья', 'кодекс', 'право', 'норма',
            'суд', 'дело', 'решение', 'постановление', 'указ'
        ]
        
        question_lower = question.lower()
        chunk_lower = chunk.lower()
        
        keyword_matches = sum(
            1 for kw in legal_keywords
            if kw in question_lower and kw in chunk_lower
        )
        
        keyword_features = np.array([
            keyword_matches,
            sum(1 for kw in legal_keywords if kw in question_lower),
            sum(1 for kw in legal_keywords if kw in chunk_lower)
        ])
        
        # Объединение всех признаков
        all_features = np.concatenate([
            overlap_features,
            length_features,
            stat_features,
            ngram_features,
            keyword_features
        ])
        
        # Дополняем частотами токенов до нужного размера
        remaining_size = self.embedding_dim - len(all_features)
        if remaining_size > 0:
            # Используем разность частот как дополнительные признаки
            freq_diff = np.abs(question_freq - chunk_freq)[:remaining_size]
            if len(freq_diff) < remaining_size:
                freq_diff = np.pad(freq_diff, (0, remaining_size - len(freq_diff)))
            all_features = np.concatenate([all_features, freq_diff])
        
        # Обрезаем или дополняем до embedding_dim
        if len(all_features) > self.embedding_dim:
            all_features = all_features[:self.embedding_dim]
        elif len(all_features) < self.embedding_dim:
            all_features = np.pad(all_features, (0, self.embedding_dim - len(all_features)))
        
        return all_features
    
    def train(
        self,
        questions: List[str],
        chunks: List[str],
        labels: List[int],  # 0 = нерелевантен, 1 = релевантен
        test_size: float = 0.2,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Обучение классификатора релевантности
        
        Args:
            questions: Список вопросов
            chunks: Список чанков
            labels: Список меток (0/1)
            test_size: Доля тестовой выборки
            n_estimators: Количество деревьев
            max_depth: Максимальная глубина деревьев
            verbose: Вывод информации
            
        Returns:
            Словарь с метриками качества
        """
        if len(questions) != len(chunks) or len(questions) != len(labels):
            raise ValueError("questions, chunks, and labels must have the same length")
        
        if verbose:
            print(f"Extracting features from {len(questions)} pairs...")
        
        # Извлечение признаков
        X = np.array([
            self._extract_features(q, c)
            for q, c in zip(questions, chunks)
        ])
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
            print(f"Train distribution: {np.bincount(y_train)}")
            print(f"Test distribution: {np.bincount(y_test)}")
        
        # Нормализация
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if verbose:
            print(f"Training {self.classifier_type} classifier...")
        
        # Обучение
        if self.classifier_type == "xgboost" and HAS_XGB:
            self.classifier = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth or 6,
                learning_rate=0.1,
                scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),  # Балансировка классов
                eval_metric='logloss',
                random_state=42
            )
        else:
            self.classifier = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight='balanced',  # Балансировка классов
                random_state=42,
                n_jobs=-1
            )
        
        self.classifier.fit(X_train_scaled, y_train)
        
        # Оценка
        y_pred = self.classifier.predict(X_test_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        if verbose:
            print(f"\nClassification Results:")
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1 Score:  {metrics['f1']:.4f}")
            print(f"\nDetailed Report:")
            print(classification_report(
                y_test, y_pred,
                target_names=['Not Relevant', 'Relevant'],
                zero_division=0
            ))
        
        return metrics
    
    def predict(self, question: str, chunk: str) -> int:
        """
        Предсказание релевантности
        
        Args:
            question: Текст вопроса
            chunk: Текст чанка
            
        Returns:
            0 (нерелевантен) или 1 (релевантен)
        """
        if self.classifier is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")
        
        features = self._extract_features(question, chunk).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        prediction = self.classifier.predict(features_scaled)[0]
        
        return int(prediction)
    
    def predict_proba(self, question: str, chunk: str) -> float:
        """
        Предсказание вероятности релевантности
        
        Args:
            question: Текст вопроса
            chunk: Текст чанка
            
        Returns:
            Вероятность релевантности [0, 1]
        """
        if self.classifier is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")
        
        features = self._extract_features(question, chunk).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        proba = self.classifier.predict_proba(features_scaled)[0]
        
        # Возвращаем вероятность класса 1 (релевантен)
        return float(proba[1])
    
    def filter_chunks(
        self,
        question: str,
        chunks: List[str],
        threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Фильтрация чанков по релевантности
        
        Args:
            question: Текст вопроса
            chunks: Список чанков
            threshold: Порог вероятности для фильтрации
            
        Returns:
            Список пар (чанк, вероятность) для релевантных чанков
        """
        if self.classifier is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")
        
        relevant_chunks = []
        
        for chunk in chunks:
            proba = self.predict_proba(question, chunk)
            if proba >= threshold:
                relevant_chunks.append((chunk, proba))
        
        # Сортируем по убыванию вероятности
        relevant_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return relevant_chunks
    
    def save_model(self, path: str):
        """Сохранение модели"""
        if self.classifier is None:
            raise ValueError("No model to save. Train the model first.")
        
        bundle = {
            'classifier': self.classifier,
            'scaler': self.scaler,
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
        self.classifier_type = bundle.get('classifier_type', 'random_forest')
        self.embedding_dim = bundle.get('embedding_dim', 256)
        
        print(f"Model loaded from {path}")
