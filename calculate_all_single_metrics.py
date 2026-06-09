import os
import subprocess
from pathlib import Path
import argparse

# python calculate_all_single_metrics.py --method npo --model gpt2_xl --percent_blocks_dropped 0 --percent_blocks_freezed 0 --unlearn_data_count 6
parser = argparse.ArgumentParser(description="calculate extended info for single forgotten facts")
parser.add_argument("--method", type=str, default="npo")
parser.add_argument("--model", type=str, default="gpt2_xl")
parser.add_argument("--percent_blocks_dropped", type=int, default=0)
parser.add_argument("--percent_blocks_freezed", type=int, default=0)
parser.add_argument("--unlearn_data_count", type=int, default=21)
parser.add_argument("--base_input_dir", type=str, default="/content/drive/MyDrive/Unlearning/unlearning_checkpoint")
parser.add_argument("--base_output_dir", type=str, default="/content/drive/MyDrive/Unlearning/results")
args = parser.parse_args()

base_input_dir = os.path.join(
    args.base_input_dir,
    args.method,
    args.model,
    f"{args.percent_blocks_dropped}_dropped",
    f"{args.percent_blocks_freezed}_freezed"
)

base_output_dir = os.path.join(
    args.base_output_dir,
    args.method,
    args.model,
    f"{args.percent_blocks_dropped}_dropped",
    f"{args.percent_blocks_freezed}_freezed",
)

Path(base_output_dir).mkdir(parents=True, exist_ok=True)

# Обходим все unlearn_data_id
for unlearn_data_id in range(args.unlearn_data_count):
    # Формируем путь к входной директории
    input_dir = f"{base_input_dir}/{unlearn_data_id}"
    
    # Проверяем существует ли директория
    if not os.path.exists(input_dir):
        print(f"Пропускаю unlearn_data_id={unlearn_data_id} - директория не найдена: {input_dir}")
        continue
    
    # Обходим чекпоинты 
    # for checkpoint in [f'checkpoint-{num}' for num in np.arange(5,8,1)]:
    # for checkpoint in [f'checkpoint-{num}' for num in [8, 16, 32]]:
    for checkpoint in os.listdir(input_dir):
        if checkpoint.startswith("checkpoint-"):
            checkpoint_path = os.path.join(input_dir, checkpoint)
            
            # Проверяем наличие необходимых файлов
            if not (
                os.path.exists(f"{checkpoint_path}/relationships_correct.pt")
                and os.path.exists(f"{checkpoint_path}/biographies_correct.pt")
            ):
                print(f"Пропускаю {checkpoint_path} - отсутствуют необходимые файлы")
                continue # переход к следующему чекпоинту

            print(f"Обработка: unlearn_data_id={unlearn_data_id}, {checkpoint}")
            
            target_dir = f"{base_output_dir}/{unlearn_data_id}/{checkpoint}"
            Path(target_dir).mkdir(parents=True, exist_ok=True)

            target_file = f"{target_dir}/rec_acc_extended.pt"
            if os.path.exists(target_file):
                print(f"Пропускаю: файл уже существует: {target_file}")
                continue
            # Сохранение rec_acc.pt, rec_acc_extended.pt
            cmd = [
                "python", "single_fact_recall_and_acc.py",
                "--unlearn_data_id", str(unlearn_data_id),
                "--input_dir", checkpoint_path,
                "--output_dir", target_dir
            ]
            # чтобы вызвать single_fact_recall_and_acc.py
            result = subprocess.run(cmd, capture_output=True, text=True)
                
            if result.returncode == 0:
                if result.stdout:
                    print(result.stdout)
                print(f"Расширенная информация сохранена в: {target_file}")
            else:
                print(f"Ошибка при обработке {checkpoint_path}:")
                print(result.stderr)


print("Обработка завершена!")