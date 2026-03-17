"""
Скрипт для конвертации времени в submission.json из миллисекунд в секунды
Делит ttft_ms, tpot_ms, total_time_ms на 1000 и отсекает дробную часть
"""

import json

def fix_times(input_file, output_file):
    """
    Конвертирует время из миллисекунд в секунды
    
    Args:
        input_file: путь к исходному файлу submission.json
        output_file: путь к новому файлу
    """
    # Читаем исходный файл
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Счетчики для статистики
    total_questions = len(data.get('answers', []))
    fixed_count = 0
    
    # Обрабатываем каждый ответ
    for answer in data.get('answers', []):
        telemetry = answer.get('telemetry', {})
        
        # Конвертируем ttft_ms
        if 'ttft_ms' in telemetry:
            old_value = telemetry['ttft_ms']
            telemetry['ttft_ms'] = int(old_value / 1000)
            fixed_count += 1
        
        # Конвертируем tpot_ms
        if 'tpot_ms' in telemetry:
            old_value = telemetry['tpot_ms']
            telemetry['tpot_ms'] = int(old_value / 1000)
        
        # Конвертируем total_time_ms
        if 'total_time_ms' in telemetry:
            old_value = telemetry['total_time_ms']
            telemetry['total_time_ms'] = int(old_value / 1000)
    
    # Сохраняем в новый файл
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Обработано вопросов: {total_questions}")
    print(f"✓ Исправлено записей: {fixed_count}")
    print(f"✓ Результат сохранен в: {output_file}")


if __name__ == "__main__":
    input_file = "hack/submission.json"
    output_file = "hack/submission_fixed.json"
    
    print("=" * 60)
    print("Конвертация времени в submission.json")
    print("=" * 60)
    print(f"\nИсходный файл: {input_file}")
    print(f"Новый файл: {output_file}")
    print(f"\nКонвертация: миллисекунды → секунды (целые числа)")
    print()
    
    try:
        fix_times(input_file, output_file)
        print("\n✓ Готово!")
    except FileNotFoundError:
        print(f"\n✗ Ошибка: файл {input_file} не найден")
    except Exception as e:
        print(f"\n✗ Ошибка: {e}")
