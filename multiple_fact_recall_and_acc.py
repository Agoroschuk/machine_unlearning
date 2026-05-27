import argparse
import os
from pathlib import Path

import numpy as np
import torch


# python multiple_fact_recall_and_acc.py --method npo --model gpt2_xl --percent_blocks_dropped 0 --percent_blocks_freezed 25 --unlearn_data_count 31
parser = argparse.ArgumentParser(description="calculate recall and accuracy for multiple forgotten facts")
parser.add_argument("--method", type=str, default="npo")
parser.add_argument("--model", type=str, default="gpt2_xl")
parser.add_argument("--percent_blocks_dropped", type=int, default=0)
parser.add_argument("--percent_blocks_freezed", type=int, default=25)
parser.add_argument("--unlearn_data_count", type=int, default=31)
parser.add_argument("--base_input_dir", type=str, default="/content/drive/MyDrive/Unlearning/miscellaneous/unlearning_checkpoint")
parser.add_argument("--base_output_dir", type=str, default="/content/drive/MyDrive/Unlearning/miscellaneous/results")
args = parser.parse_args()


input_dir = os.path.join(
    args.base_input_dir,
    args.method,
    args.model,
    f"{args.percent_blocks_dropped}_dropped",
    f"{args.percent_blocks_freezed}_freezed",
    f"first_{args.unlearn_data_count}",
)

output_dir = os.path.join(
    args.base_output_dir,
    args.method,
    args.model,
    f"{args.percent_blocks_dropped}_dropped",
    f"{args.percent_blocks_freezed}_freezed",
    f"first_{args.unlearn_data_count}",
)

print("input_dir", input_dir)
print("output_dir", output_dir)

Path(output_dir).mkdir(parents=True, exist_ok=True)

subsample = torch.load("synthetic_data/subsample.pt", weights_only=False)
ids_rel_to_unlearn = [int(x) for x in subsample[:args.unlearn_data_count]]
ids_rel_to_unlearn_set = set(ids_rel_to_unlearn)

if not os.path.exists(input_dir):
    raise FileNotFoundError(f"Input directory not found: {input_dir}")


for checkpoint in sorted(os.listdir(input_dir)):
    if checkpoint.startswith("checkpoint-"):
        checkpoint_path = os.path.join(input_dir, checkpoint)

        if (
            os.path.exists(f"{checkpoint_path}/relationships_correct.pt")
            and os.path.exists(f"{checkpoint_path}/biographies_correct.pt")
        ):
            print(f"Обработка: first={args.unlearn_data_count}, {checkpoint}")

            # True <=> correct answer was given after forgetting
            rel_correct = torch.load(f"{checkpoint_path}/relationships_correct.pt", weights_only=False)
            bio_correct = torch.load(f"{checkpoint_path}/biographies_correct.pt", weights_only=False)

            rel_correct = np.asarray(rel_correct).astype(bool)
            bio_correct = np.asarray(bio_correct).astype(bool)

            # list of fact ids to retain (facts to forget are excluded via not in ids_rel_to_unlearn_set)
            ids_rel_to_retain = [
                data_id for data_id in range(len(rel_correct))
                if data_id not in ids_rel_to_unlearn_set
            ]

            # get 0(retained) or 1(forgotten) status for forget facts
            forgotten_rel = np.asarray([
                not rel_correct[data_id]
                for data_id in ids_rel_to_unlearn
            ])
            
            # get 1(retained) or 0(forgotten) status for retain facts
            retained_rel = np.asarray([
                rel_correct[data_id]
                for data_id in ids_rel_to_retain
            ])

            retained_bio = bio_correct

            # fraction of correctly forgotten from unlearn_data_count relationships
            recall = forgotten_rel.sum() / len(forgotten_rel)

            # fraction of correctly saved from retained relationships, ideally 1
            acc_rel = retained_rel.sum() / len(retained_rel)

            # fraction of correctly saved from biographies, ideally 1
            acc_bio = retained_bio.sum() / len(retained_bio)

            # mixed accuracy over retained relationships and all biographies
            acc_all = (retained_rel.sum() + retained_bio.sum()) / (len(retained_rel) + len(retained_bio))

            print(("recall", "accuracy of relationships", "accuracy of biographies", "accuracy of all knowledge base"))
            print((recall, acc_rel, acc_bio, acc_all))

            target_dir = os.path.join(output_dir, checkpoint)
            Path(target_dir).mkdir(parents=True, exist_ok=True)

            target_file = os.path.join(target_dir, "rec_acc.pt")
            if not os.path.exists(target_file):
                torch.save((recall, acc_rel, acc_bio, acc_all), target_file)
                print(f"Result saved to: {target_file}")
            else:
                print(f"File with result already exists: {target_file}")
        else:
            print(f"Missing {checkpoint_path} - no required files")

print("Processing finished!")