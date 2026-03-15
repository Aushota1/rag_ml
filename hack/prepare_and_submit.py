#!/usr/bin/env python3
"""
Мастер-скрипт для подготовки и отправки решения
Выполняет все шаги автоматически
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Выполнить команду и показать результат"""
    print("\n" + "=" * 60)
    print(f"▶ {description}")
    print("=" * 60)
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Ошибка при выполнении: {description}")
        return False
    
    print(f"\n✓ {description} - завершено")
    return True

def main():
    print("=" * 60)
    print("ПОДГОТОВКА И ОТПРАВКА РЕШЕНИЯ НА ХАКАТОН")
    print("=" * 60)
    
    # Проверяем, что мы в правильной директории
    hack_dir = Path(__file__).parent
    os.chdir(hack_dir)
    
    # Шаг 1: Генерация ответов
    if not run_command(
        f"{sys.executable} generate_submission.py",
        "Генерация submission.json"
    ):
        return 1
    
    # Шаг 2: Создание архива
    if not run_command(
        f"{sys.executable} create_archive.py",
        "Создание code_archive.zip"
    ):
        return 1
    
    # Шаг 3: Проверка файлов
    print("\n" + "=" * 60)
    print("▶ Проверка созданных файлов")
    print("=" * 60)
    
    submission_path = hack_dir / "submission.json"
    archive_path = hack_dir / "code_archive.zip"
    
    if not submission_path.exists():
        print("❌ Файл submission.json не найден!")
        return 1
    
    if not archive_path.exists():
        print("❌ Файл code_archive.zip не найден!")
        return 1
    
    submission_size = submission_path.stat().st_size / 1024
    archive_size = archive_path.stat().st_size / (1024 * 1024)
    
    print(f"\n✓ submission.json: {submission_size:.2f} KB")
    print(f"✓ code_archive.zip: {archive_size:.2f} MB")
    
    # Шаг 4: Отправка (опционально)
    print("\n" + "=" * 60)
    print("▶ Отправка на платформу")
    print("=" * 60)
    
    api_key = os.environ.get('HACKATHON_API_KEY')
    
    if not api_key:
        print("\n⚠️  API ключ не установлен!")
        print("\nДля отправки решения:")
        print("  1. Установите API ключ:")
        print("     export HACKATHON_API_KEY='your-api-key'  # Linux/Mac")
        print("     set HACKATHON_API_KEY=your-api-key       # Windows")
        print("  2. Запустите:")
        print("     bash submit.sh      # Linux/Mac")
        print("     submit.bat          # Windows")
        print("\nИли отправьте вручную через curl:")
        print('  curl -X POST "https://platform.agentic-challenge.ai/api/v1/submissions" \\')
        print('    -H "X-API-Key: YOUR_KEY" \\')
        print('    -F "file=@./submission.json;type=application/json" \\')
        print('    -F "code_archive=@./code_archive.zip;type=application/zip"')
    else:
        print(f"\n✓ API ключ найден: {api_key[:8]}...")
        
        response = input("\nОтправить решение сейчас? (y/n): ")
        if response.lower() == 'y':
            # Определяем ОС и запускаем соответствующий скрипт
            if sys.platform == 'win32':
                if not run_command("submit.bat", "Отправка решения"):
                    return 1
            else:
                if not run_command("bash submit.sh", "Отправка решения"):
                    return 1
        else:
            print("\n⚠️  Отправка пропущена")
            print("Для отправки позже запустите:")
            print("  bash submit.sh      # Linux/Mac")
            print("  submit.bat          # Windows")
    
    # Итоги
    print("\n" + "=" * 60)
    print("✓ ПОДГОТОВКА ЗАВЕРШЕНА!")
    print("=" * 60)
    print("\nСозданные файлы:")
    print(f"  📄 {submission_path}")
    print(f"  📦 {archive_path}")
    print("\nСледующие шаги:")
    print("  1. Проверьте submission.json")
    print("  2. Проверьте code_archive.zip")
    print("  3. Отправьте на платформу (если еще не отправлено)")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
