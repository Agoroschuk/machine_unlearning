import os
import subprocess
import shutil
from pathlib import Path

method = 'ga'
# method = 'npo'
model = 'gpt2_xl'
base_input_dir = f"/content/drive/MyDrive/Unlearning/miscellaneous/unlearning_checkpoint/{method}/{model}"
base_output_dir = f"/content/drive/MyDrive/Unlearning/miscellaneous/results/{method}/{model}"
print('base_input_dir', base_input_dir)
print('base_output_dir', base_output_dir)

# Создаем базовую выходную директорию
Path(base_output_dir).mkdir(parents=True, exist_ok=True)

# Обходим все unlearn_data_id
for unlearn_data_id in range(0, 31):
    # Формируем путь к входной директории
    input_dir = f"{base_input_dir}/{unlearn_data_id}"
    
    # Проверяем существует ли директория
    if not os.path.exists(input_dir):
        print(f"Пропускаю unlearn_data_id={unlearn_data_id} - директория не найдена: {input_dir}")
        continue
    
    # Обходим все чекпоинты
    for checkpoint in os.listdir(input_dir):
        if checkpoint.startswith("checkpoint-"):
            checkpoint_path = os.path.join(input_dir, checkpoint)
            
            # Проверяем наличие необходимых файлов
            if (os.path.exists(f"{checkpoint_path}/relationships_correct.pt") and 
                os.path.exists(f"{checkpoint_path}/biographies_correct.pt")):
                
                print(f"Обрабатываю: unlearn_data_id={unlearn_data_id}, {checkpoint}")
                
                # Запускаем оригинальный скрипт, в его результате rec_acc.pt сохраняются в директории чекпоинтов с ответами моделей
                cmd = [
                    "python", "calculate_recall_and_acc.py",
                    "--unlearn_data_id", str(unlearn_data_id),
                    "--input_dir", checkpoint_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Создаем целевую директорию
                    target_dir = f"{base_output_dir}/{unlearn_data_id}/{checkpoint}"
                    Path(target_dir).mkdir(parents=True, exist_ok=True)
                    
                    # Перемещаем результат в целевую директорию, т.к. сначала rec_acc.pt лежит в папке с чекпоинтами
                    source_file = f"{checkpoint_path}/rec_acc.pt"
                    # if os.path.exists(source_file):
                    if os.path.exists(source_file) and not os.path.exists(f"{target_dir}/rec_acc.pt"):
                        shutil.copy2(source_file, f"{target_dir}/rec_acc.pt")
                        print(f"Результат сохранен в: {target_dir}/rec_acc.pt")
                        os.remove(source_file)
                    else:
                        # удаление на случай, что файл был создан ранее
                        os.remove(source_file)
                        print(f"Ошибка: файл результатов не создан: {source_file}")
                else:
                    print(f"Ошибка при обработке {checkpoint_path}:")
                    print(result.stderr)
            else:
                print(f"Пропускаю {checkpoint_path} - отсутствуют необходимые файлы")

print("Обработка завершена!")