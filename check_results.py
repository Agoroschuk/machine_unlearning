import os
import torch

base_dir = "/content/drive/MyDrive/Unlearning/models/results/ga/gpt2_xl"

print("Проверка результатов:")
for unlearn_data_id in range(55):
    unlearn_dir = f"{base_dir}/{unlearn_data_id}"
    if os.path.exists(unlearn_dir):
        checkpoints = os.listdir(unlearn_dir)
        if checkpoints:
            print(f"unlearn_data_id={unlearn_data_id}: {len(checkpoints)} чекпоинтов")
            for checkpoint in checkpoints:
                result_file = f"{unlearn_dir}/{checkpoint}/rec_acc.pt"
                if os.path.exists(result_file):
                    try:
                        metrics = torch.load(result_file)
                        print(f"  {checkpoint}: recall={metrics[0]:.3f}, acc_rel={metrics[1]:.3f}, acc_bio={metrics[2]:.3f}, acc_all={metrics[3]:.3f}")
                    except:
                        print(f"  {checkpoint}: ошибка чтения файла")