"""
3D визуализация embedding модели
Создает интерактивные графики для анализа эмбеддингов
"""

import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from typing import Optional, List, Tuple
import pickle


class EmbeddingVisualizer:
    """Визуализатор для embedding модели"""
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: Путь к файлу модели (.pth)
        """
        self.model_path = Path(model_path)
        self.model_data = None
        self.embeddings = None
        self.vocab_size = None
        self.embedding_dim = None
        
        self._load_model()
    
    def _load_model(self):
        """Загрузка модели"""
        print(f"Loading model from: {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Загружаем модель
        self.model_data = torch.load(self.model_path, map_location='cpu')
        
        # Извлекаем embeddings
        if isinstance(self.model_data, dict):
            # Проверяем разные возможные структуры
            if 'embedding.weight' in self.model_data:
                self.embeddings = self.model_data['embedding.weight'].numpy()
            elif 'state_dict' in self.model_data:
                # Ищем в state_dict
                state_dict = self.model_data['state_dict']
                if isinstance(state_dict, dict):
                    # Ищем embedding слой
                    for key, value in state_dict.items():
                        if isinstance(value, torch.Tensor) and len(value.shape) == 2:
                            if 'embedding' in key.lower() and 'weight' in key:
                                self.embeddings = value.numpy()
                                print(f"  Found embeddings at: {key}")
                                break
            elif 'model_state_dict' in self.model_data:
                state_dict = self.model_data['model_state_dict']
                if 'embedding.weight' in state_dict:
                    self.embeddings = state_dict['embedding.weight'].numpy()
                else:
                    # Ищем любой embedding слой
                    for key, value in state_dict.items():
                        if isinstance(value, torch.Tensor) and len(value.shape) == 2:
                            if 'embedding' in key.lower() and 'weight' in key:
                                self.embeddings = value.numpy()
                                print(f"  Found embeddings at: {key}")
                                break
            
            # Если не нашли, пробуем найти первый 2D тензор
            if self.embeddings is None:
                for key, value in self.model_data.items():
                    if isinstance(value, torch.Tensor) and len(value.shape) == 2:
                        self.embeddings = value.numpy()
                        print(f"  Found 2D tensor at: {key}")
                        break
        elif isinstance(self.model_data, torch.Tensor):
            self.embeddings = self.model_data.numpy()
        
        if self.embeddings is None:
            raise ValueError("Could not find embeddings in model file")
        
        self.vocab_size, self.embedding_dim = self.embeddings.shape
        
        print(f"✓ Model loaded successfully")
        print(f"  Vocabulary size: {self.vocab_size:,}")
        print(f"  Embedding dimension: {self.embedding_dim}")
    
    def reduce_dimensions(self, method: str = 'umap', n_components: int = 3, 
                         sample_size: Optional[int] = None) -> np.ndarray:
        """
        Снижение размерности эмбеддингов
        
        Args:
            method: Метод снижения размерности ('pca', 'tsne', 'umap')
            n_components: Количество компонент (2 или 3)
            sample_size: Количество токенов для визуализации (None = все)
            
        Returns:
            Массив с пониженной размерностью
        """
        # Сэмплирование если нужно
        if sample_size and sample_size < self.vocab_size:
            indices = np.random.choice(self.vocab_size, sample_size, replace=False)
            embeddings = self.embeddings[indices]
        else:
            embeddings = self.embeddings
            indices = np.arange(self.vocab_size)
        
        print(f"\nReducing dimensions using {method.upper()}...")
        print(f"  Input shape: {embeddings.shape}")
        print(f"  Target dimensions: {n_components}D")
        
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(embeddings)
            variance = reducer.explained_variance_ratio_
            print(f"  Explained variance: {variance.sum():.2%}")
            
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, 
                          perplexity=min(30, len(embeddings) - 1))
            reduced = reducer.fit_transform(embeddings)
            
        elif method == 'umap':
            reducer = umap.UMAP(n_components=n_components, random_state=42,
                               n_neighbors=min(15, len(embeddings) - 1))
            reduced = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"  Output shape: {reduced.shape}")
        
        return reduced, indices
    
    def plot_3d_scatter(self, method: str = 'umap', sample_size: int = 5000,
                       output_file: Optional[str] = None):
        """
        3D scatter plot эмбеддингов
        
        Args:
            method: Метод снижения размерности
            sample_size: Количество точек
            output_file: Путь для сохранения HTML
        """
        # Снижаем размерность
        reduced, indices = self.reduce_dimensions(method, n_components=3, 
                                                  sample_size=sample_size)
        
        # Вычисляем нормы для цветовой шкалы
        norms = np.linalg.norm(self.embeddings[indices], axis=1)
        
        # Создаем 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=reduced[:, 0],
            y=reduced[:, 1],
            z=reduced[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=norms,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Norm"),
                opacity=0.8
            ),
            text=[f"Token {i}<br>Norm: {norms[j]:.3f}" 
                  for j, i in enumerate(indices)],
            hovertemplate='<b>%{text}</b><br>' +
                         'X: %{x:.3f}<br>' +
                         'Y: %{y:.3f}<br>' +
                         'Z: %{z:.3f}<br>' +
                         '<extra></extra>'
        )])
        
        fig.update_layout(
            title=f'3D Embedding Visualization ({method.upper()})<br>' +
                  f'<sub>Vocab: {self.vocab_size:,} tokens, Dim: {self.embedding_dim}, Sample: {len(indices):,}</sub>',
            scene=dict(
                xaxis_title=f'{method.upper()} 1',
                yaxis_title=f'{method.upper()} 2',
                zaxis_title=f'{method.upper()} 3',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1200,
            height=800,
            hovermode='closest'
        )
        
        if output_file:
            fig.write_html(output_file)
            print(f"\n✓ Saved to: {output_file}")
        
        return fig
    
    def plot_2d_scatter(self, method: str = 'umap', sample_size: int = 10000,
                       output_file: Optional[str] = None):
        """
        2D scatter plot эмбеддингов
        
        Args:
            method: Метод снижения размерности
            sample_size: Количество точек
            output_file: Путь для сохранения HTML
        """
        # Снижаем размерность
        reduced, indices = self.reduce_dimensions(method, n_components=2, 
                                                  sample_size=sample_size)
        
        # Вычисляем нормы
        norms = np.linalg.norm(self.embeddings[indices], axis=1)
        
        # Создаем 2D scatter plot
        fig = go.Figure(data=[go.Scatter(
            x=reduced[:, 0],
            y=reduced[:, 1],
            mode='markers',
            marker=dict(
                size=5,
                color=norms,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Norm"),
                opacity=0.7
            ),
            text=[f"Token {i}<br>Norm: {norms[j]:.3f}" 
                  for j, i in enumerate(indices)],
            hovertemplate='<b>%{text}</b><br>' +
                         'X: %{x:.3f}<br>' +
                         'Y: %{y:.3f}<br>' +
                         '<extra></extra>'
        )])
        
        fig.update_layout(
            title=f'2D Embedding Visualization ({method.upper()})<br>' +
                  f'<sub>Vocab: {self.vocab_size:,} tokens, Dim: {self.embedding_dim}, Sample: {len(indices):,}</sub>',
            xaxis_title=f'{method.upper()} 1',
            yaxis_title=f'{method.upper()} 2',
            width=1200,
            height=800,
            hovermode='closest'
        )
        
        if output_file:
            fig.write_html(output_file)
            print(f"\n✓ Saved to: {output_file}")
        
        return fig
    
    def plot_distribution_analysis(self, output_file: Optional[str] = None):
        """
        Анализ распределения эмбеддингов
        
        Args:
            output_file: Путь для сохранения HTML
        """
        # Вычисляем статистики
        norms = np.linalg.norm(self.embeddings, axis=1)
        means = np.mean(self.embeddings, axis=1)
        stds = np.std(self.embeddings, axis=1)
        
        # Создаем subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Distribution of Embedding Norms',
                'Distribution of Embedding Means',
                'Distribution of Embedding Stds',
                'Norm vs Mean'
            ),
            specs=[[{'type': 'histogram'}, {'type': 'histogram'}],
                   [{'type': 'histogram'}, {'type': 'scatter'}]]
        )
        
        # Histogram нормы
        fig.add_trace(
            go.Histogram(x=norms, nbinsx=50, name='Norms',
                        marker_color='blue', opacity=0.7),
            row=1, col=1
        )
        
        # Histogram средних
        fig.add_trace(
            go.Histogram(x=means, nbinsx=50, name='Means',
                        marker_color='green', opacity=0.7),
            row=1, col=2
        )
        
        # Histogram стандартных отклонений
        fig.add_trace(
            go.Histogram(x=stds, nbinsx=50, name='Stds',
                        marker_color='red', opacity=0.7),
            row=2, col=1
        )
        
        # Scatter norm vs mean
        fig.add_trace(
            go.Scatter(x=means, y=norms, mode='markers',
                      marker=dict(size=3, opacity=0.5),
                      name='Norm vs Mean'),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'Embedding Distribution Analysis<br>' +
                  f'<sub>Vocab: {self.vocab_size:,} tokens, Dim: {self.embedding_dim}</sub>',
            showlegend=False,
            width=1400,
            height=1000
        )
        
        if output_file:
            fig.write_html(output_file)
            print(f"\n✓ Saved to: {output_file}")
        
        return fig
    
    def plot_dimension_analysis(self, output_file: Optional[str] = None):
        """
        Анализ по измерениям
        
        Args:
            output_file: Путь для сохранения HTML
        """
        # Статистики по измерениям
        dim_means = np.mean(self.embeddings, axis=0)
        dim_stds = np.std(self.embeddings, axis=0)
        dim_mins = np.min(self.embeddings, axis=0)
        dim_maxs = np.max(self.embeddings, axis=0)
        
        dimensions = np.arange(self.embedding_dim)
        
        # Создаем subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Mean per Dimension',
                'Std per Dimension',
                'Min/Max per Dimension',
                'Dimension Variance'
            )
        )
        
        # Mean
        fig.add_trace(
            go.Scatter(x=dimensions, y=dim_means, mode='lines',
                      name='Mean', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Std
        fig.add_trace(
            go.Scatter(x=dimensions, y=dim_stds, mode='lines',
                      name='Std', line=dict(color='green')),
            row=1, col=2
        )
        
        # Min/Max
        fig.add_trace(
            go.Scatter(x=dimensions, y=dim_mins, mode='lines',
                      name='Min', line=dict(color='red')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=dimensions, y=dim_maxs, mode='lines',
                      name='Max', line=dict(color='orange')),
            row=2, col=1
        )
        
        # Variance
        dim_vars = np.var(self.embeddings, axis=0)
        fig.add_trace(
            go.Bar(x=dimensions, y=dim_vars, name='Variance',
                  marker_color='purple'),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'Dimension-wise Analysis<br>' +
                  f'<sub>Vocab: {self.vocab_size:,} tokens, Dim: {self.embedding_dim}</sub>',
            showlegend=True,
            width=1400,
            height=1000
        )
        
        if output_file:
            fig.write_html(output_file)
            print(f"\n✓ Saved to: {output_file}")
        
        return fig
    
    def create_full_report(self, output_dir: str = "visualizations"):
        """
        Создает полный отчет со всеми визуализациями
        
        Args:
            output_dir: Директория для сохранения
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("=" * 70)
        print("CREATING FULL VISUALIZATION REPORT")
        print("=" * 70)
        
        # 3D UMAP
        print("\n1. Creating 3D UMAP visualization...")
        self.plot_3d_scatter(
            method='umap',
            sample_size=5000,
            output_file=str(output_path / "3d_umap.html")
        )
        
        # 3D PCA
        print("\n2. Creating 3D PCA visualization...")
        self.plot_3d_scatter(
            method='pca',
            sample_size=5000,
            output_file=str(output_path / "3d_pca.html")
        )
        
        # 2D UMAP
        print("\n3. Creating 2D UMAP visualization...")
        self.plot_2d_scatter(
            method='umap',
            sample_size=10000,
            output_file=str(output_path / "2d_umap.html")
        )
        
        # Distribution analysis
        print("\n4. Creating distribution analysis...")
        self.plot_distribution_analysis(
            output_file=str(output_path / "distribution_analysis.html")
        )
        
        # Dimension analysis
        print("\n5. Creating dimension analysis...")
        self.plot_dimension_analysis(
            output_file=str(output_path / "dimension_analysis.html")
        )
        
        print("\n" + "=" * 70)
        print("✅ REPORT COMPLETE!")
        print("=" * 70)
        print(f"\nAll visualizations saved to: {output_path.absolute()}")
        print("\nGenerated files:")
        print("  - 3d_umap.html              (3D UMAP projection)")
        print("  - 3d_pca.html               (3D PCA projection)")
        print("  - 2d_umap.html              (2D UMAP projection)")
        print("  - distribution_analysis.html (Statistical distributions)")
        print("  - dimension_analysis.html    (Per-dimension analysis)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize embedding model in 3D"
    )
    parser.add_argument(
        '--model',
        default='models/tokenier/embedding_model.pth',
        help='Path to embedding model (.pth file)'
    )
    parser.add_argument(
        '--output-dir',
        default='visualizations',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--method',
        choices=['umap', 'pca', 'tsne'],
        default='umap',
        help='Dimensionality reduction method'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=5000,
        help='Number of tokens to visualize'
    )
    parser.add_argument(
        '--full-report',
        action='store_true',
        help='Create full report with all visualizations'
    )
    parser.add_argument(
        '--2d',
        action='store_true',
        help='Create 2D visualization instead of 3D'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("EMBEDDING MODEL VISUALIZATION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    
    # Создаем визуализатор
    visualizer = EmbeddingVisualizer(args.model)
    
    if args.full_report:
        # Полный отчет
        visualizer.create_full_report(args.output_dir)
    else:
        # Одна визуализация
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if args.__dict__['2d']:
            output_file = output_path / f"2d_{args.method}.html"
            visualizer.plot_2d_scatter(
                method=args.method,
                sample_size=args.sample_size,
                output_file=str(output_file)
            )
        else:
            output_file = output_path / f"3d_{args.method}.html"
            visualizer.plot_3d_scatter(
                method=args.method,
                sample_size=args.sample_size,
                output_file=str(output_file)
            )
        
        print("\n" + "=" * 70)
        print("✅ DONE!")
        print("=" * 70)
        print(f"\nVisualization saved to: {output_file}")
        print(f"\nOpen in browser to view interactive 3D plot")


if __name__ == "__main__":
    main()
