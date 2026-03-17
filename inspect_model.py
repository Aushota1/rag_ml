"""
Инспекция структуры модели
"""

import torch
from pathlib import Path


def inspect_model(model_path: str):
    """Показывает структуру файла модели"""
    
    print("=" * 70)
    print("MODEL INSPECTION")
    print("=" * 70)
    print(f"File: {model_path}")
    
    path = Path(model_path)
    if not path.exists():
        print(f"❌ File not found: {path}")
        return
    
    print(f"Size: {path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Загружаем модель
    print("\nLoading model...")
    try:
        model_data = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print("✓ Model loaded successfully")
    
    # Анализируем структуру
    print("\n" + "=" * 70)
    print("MODEL STRUCTURE")
    print("=" * 70)
    
    print(f"\nType: {type(model_data)}")
    
    if isinstance(model_data, dict):
        print(f"\nDictionary with {len(model_data)} keys:")
        for key in model_data.keys():
            value = model_data[key]
            print(f"\n  Key: '{key}'")
            print(f"    Type: {type(value)}")
            
            if isinstance(value, torch.Tensor):
                print(f"    Shape: {value.shape}")
                print(f"    Dtype: {value.dtype}")
                print(f"    Device: {value.device}")
                print(f"    Min: {value.min().item():.6f}")
                print(f"    Max: {value.max().item():.6f}")
                print(f"    Mean: {value.mean().item():.6f}")
            elif isinstance(value, dict):
                print(f"    Nested dict with {len(value)} keys:")
                for subkey in list(value.keys())[:10]:  # Первые 10
                    subvalue = value[subkey]
                    if isinstance(subvalue, torch.Tensor):
                        print(f"      '{subkey}': Tensor{subvalue.shape}")
                    else:
                        print(f"      '{subkey}': {type(subvalue)}")
                if len(value) > 10:
                    print(f"      ... and {len(value) - 10} more")
            elif isinstance(value, (list, tuple)):
                print(f"    Length: {len(value)}")
                if len(value) > 0:
                    print(f"    First item type: {type(value[0])}")
            else:
                print(f"    Value: {str(value)[:100]}")
    
    elif isinstance(model_data, torch.Tensor):
        print("\nDirect tensor:")
        print(f"  Shape: {model_data.shape}")
        print(f"  Dtype: {model_data.dtype}")
        print(f"  Device: {model_data.device}")
        print(f"  Min: {model_data.min().item():.6f}")
        print(f"  Max: {model_data.max().item():.6f}")
        print(f"  Mean: {model_data.mean().item():.6f}")
    
    else:
        print(f"\nUnknown type: {type(model_data)}")
        print(f"Value: {str(model_data)[:200]}")
    
    # Ищем embedding слои
    print("\n" + "=" * 70)
    print("SEARCHING FOR EMBEDDINGS")
    print("=" * 70)
    
    embeddings_found = []
    
    def search_embeddings(obj, path=""):
        """Рекурсивный поиск embedding тензоров"""
        if isinstance(obj, torch.Tensor):
            if len(obj.shape) == 2:  # Embedding обычно 2D
                embeddings_found.append({
                    'path': path,
                    'shape': obj.shape,
                    'size': obj.numel()
                })
        elif isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                search_embeddings(value, new_path)
        elif isinstance(obj, (list, tuple)):
            for i, value in enumerate(obj):
                new_path = f"{path}[{i}]"
                search_embeddings(value, new_path)
    
    search_embeddings(model_data)
    
    if embeddings_found:
        print(f"\n✓ Found {len(embeddings_found)} potential embedding tensors:")
        for i, emb in enumerate(embeddings_found, 1):
            print(f"\n  {i}. Path: {emb['path']}")
            print(f"     Shape: {emb['shape']}")
            print(f"     Size: {emb['size']:,} parameters")
            print(f"     Vocab size: {emb['shape'][0]:,}")
            print(f"     Embedding dim: {emb['shape'][1]}")
    else:
        print("\n❌ No 2D tensors found (potential embeddings)")
    
    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "models/tokenier/embedding_model.pth"
    
    inspect_model(model_path)
