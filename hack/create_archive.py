#!/usr/bin/env python3
"""
Создание code_archive.zip для отправки
Упаковывает весь код проекта
"""

import zipfile
from pathlib import Path
import os

def create_code_archive():
    print("=" * 60)
    print("Создание code_archive.zip")
    print("=" * 60)
    
    # Корневая директория проекта
    project_root = Path(__file__).parent.parent
    archive_path = Path(__file__).parent / "code_archive.zip"
    
    # Файлы и папки для включения
    include_patterns = [
        "*.py",
        "*.md",
        "*.txt",
        "*.json",
        "*.html",
        ".env.example",
        ".gitignore"
    ]
    
    # Папки для исключения
    exclude_dirs = {
        "__pycache__",
        ".git",
        ".vscode",
        "index",  # Индекс не включаем, он большой
        "models",  # Модели не включаем, они большие
        "hack"  # Папку hack не включаем в архив
    }
    
    files_to_archive = []
    
    # Собираем файлы
    for pattern in include_patterns:
        for file_path in project_root.rglob(pattern):
            # Проверяем, что файл не в исключенных директориях
            relative_path = file_path.relative_to(project_root)
            if not any(excluded in relative_path.parts for excluded in exclude_dirs):
                files_to_archive.append((file_path, relative_path))
    
    print(f"\nНайдено файлов: {len(files_to_archive)}")
    
    # Создаем архив
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path, relative_path in sorted(files_to_archive):
            print(f"  Добавление: {relative_path}")
            zipf.write(file_path, relative_path)
    
    # Проверяем размер
    size_mb = archive_path.stat().st_size / (1024 * 1024)
    
    print("\n" + "=" * 60)
    print(f"✓ Архив создан: {archive_path}")
    print(f"  Размер: {size_mb:.2f} MB")
    print(f"  Файлов: {len(files_to_archive)}")
    print("=" * 60)

if __name__ == "__main__":
    create_code_archive()
