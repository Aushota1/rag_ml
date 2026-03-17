# 🎨 3D Визуализация Embedding Модели

## Быстрый старт

### Полный отчет (рекомендуется)

Создает все визуализации сразу:

```bash
python visualize_embedding_model.py --full-report
```

Результат:
- `visualizations/3d_umap.html` - 3D UMAP проекция
- `visualizations/3d_pca.html` - 3D PCA проекция
- `visualizations/2d_umap.html` - 2D UMAP проекция
- `visualizations/distribution_analysis.html` - Статистический анализ
- `visualizations/dimension_analysis.html` - Анализ по измерениям

### Одна визуализация

```bash
# 3D UMAP (по умолчанию)
python visualize_embedding_model.py

# 3D PCA
python visualize_embedding_model.py --method pca

# 2D UMAP
python visualize_embedding_model.py --2d --method umap

# С большим количеством точек
python visualize_embedding_model.py --sample-size 10000
```

## Параметры

### Основные

- `--model` - путь к модели (по умолчанию: `models/tokenier/embedding_model.pth`)
- `--output-dir` - директория для сохранения (по умолчанию: `visualizations`)
- `--full-report` - создать все визуализации

### Настройка визуализации

- `--method` - метод снижения размерности:
  - `umap` - UMAP (рекомендуется, лучше сохраняет структуру)
  - `pca` - PCA (быстрее, линейный)
  - `tsne` - t-SNE (медленнее, хорошо для кластеров)

- `--sample-size` - количество токенов для визуализации:
  - `1000` - быстро, для теста
  - `5000` - оптимально (по умолчанию)
  - `10000` - детально, медленнее

- `--2d` - создать 2D визуализацию вместо 3D

## Что показывают визуализации

### 1. 3D UMAP/PCA/t-SNE

Интерактивная 3D проекция эмбеддингов:
- Каждая точка = токен
- Цвет = норма вектора (яркость эмбеддинга)
- Близкие точки = семантически похожие токены
- Можно вращать, зумить, наводить курсор

### 2. Distribution Analysis

Статистическое распределение:
- **Norms** - распределение норм векторов
- **Means** - распределение средних значений
- **Stds** - распределение стандартных отклонений
- **Norm vs Mean** - корреляция между нормой и средним

### 3. Dimension Analysis

Анализ по измерениям:
- **Mean per Dimension** - среднее значение каждого измерения
- **Std per Dimension** - разброс по измерениям
- **Min/Max per Dimension** - диапазон значений
- **Dimension Variance** - важность каждого измерения

## Примеры использования

### Быстрая проверка модели

```bash
python visualize_embedding_model.py --method pca --sample-size 1000
```

### Детальный анализ

```bash
python visualize_embedding_model.py --full-report --sample-size 10000
```

### Сравнение методов

```bash
# UMAP
python visualize_embedding_model.py --method umap --output-dir viz_umap

# PCA
python visualize_embedding_model.py --method pca --output-dir viz_pca

# t-SNE
python visualize_embedding_model.py --method tsne --output-dir viz_tsne
```

### Кастомная модель

```bash
python visualize_embedding_model.py \
  --model path/to/your/model.pth \
  --output-dir my_visualizations \
  --full-report
```

## Интерпретация результатов

### Хорошие признаки

✅ Равномерное распределение точек (нет больших пустот)
✅ Четкие кластеры (группы похожих токенов)
✅ Нормы векторов в разумном диапазоне (не слишком большие/маленькие)
✅ Variance распределена по измерениям (не сконцентрирована в нескольких)

### Проблемы

❌ Все точки в одном месте → модель не обучена
❌ Огромные выбросы → нестабильное обучение
❌ Вся variance в 1-2 измерениях → недоиспользование пространства
❌ Очень большие/маленькие нормы → проблемы с масштабом

## Технические детали

### Методы снижения размерности

**UMAP (Uniform Manifold Approximation and Projection)**
- Лучше всего сохраняет глобальную и локальную структуру
- Быстрее чем t-SNE
- Рекомендуется для большинства случаев

**PCA (Principal Component Analysis)**
- Самый быстрый
- Линейное преобразование
- Хорош для понимания основных направлений variance

**t-SNE (t-Distributed Stochastic Neighbor Embedding)**
- Отлично показывает кластеры
- Медленный на больших данных
- Не сохраняет глобальные расстояния

### Требования

Все зависимости уже в `requirements.txt`:
- torch
- numpy
- plotly
- scikit-learn
- umap-learn

## Troubleshooting

### Ошибка: Model not found
```bash
# Проверьте путь к модели
ls models/tokenier/embedding_model.pth
```

### Ошибка: Could not find embeddings
Модель должна содержать:
- `embedding.weight` или
- `model_state_dict['embedding.weight']` или
- Любой тензор размерности 2

### Слишком долго
- Уменьшите `--sample-size` до 1000-2000
- Используйте `--method pca` вместо umap/tsne
- Не используйте `--full-report` для быстрого теста

### Out of memory
- Уменьшите `--sample-size`
- Закройте другие программы
- Используйте 2D вместо 3D (`--2d`)

## Примеры команд

```bash
# Минимальный тест
python visualize_embedding_model.py --method pca --sample-size 500

# Стандартная визуализация
python visualize_embedding_model.py --full-report

# Максимальная детализация
python visualize_embedding_model.py --full-report --sample-size 20000

# Только 2D
python visualize_embedding_model.py --2d --method umap --sample-size 10000

# Кастомный путь
python visualize_embedding_model.py \
  --model "C:/Users/Aushota/Desktop/rag_ml/models/tokenier/embedding_model.pth" \
  --output-dir "C:/Users/Aushota/Desktop/embeddings_viz" \
  --full-report
```

## Открытие результатов

После создания визуализаций:

1. Откройте файл `.html` в браузере (Chrome, Firefox, Edge)
2. Используйте мышь для взаимодействия:
   - Левая кнопка + движение = вращение
   - Колесико = зум
   - Правая кнопка + движение = перемещение
   - Наведение курсора = информация о точке

## Что дальше?

После визуализации:
1. Проанализируйте структуру эмбеддингов
2. Проверьте качество обучения
3. Найдите проблемные области
4. Используйте для отладки модели
5. Сравните разные версии модели
