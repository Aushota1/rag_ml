#!/usr/bin/env python3
"""
Переиндексация с новым PyMuPDF парсером
"""
import os
from dotenv import load_dotenv
load_dotenv()

# Удаляем старый индекс
import shutil
from pathlib import Path

index_path = Path("index")
if index_path.exists():
    print(f"Removing old index: {index_path}")
    shutil.rmtree(index_path)
    print("✓ Old index removed\n")

# Запускаем build_index с новым парсером
print("Building new index with PyMuPDF parser...")
print("=" * 60)

import build_index
build_index.main()
