import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import argparse
import pandas as pd
from accelerate import Accelerator
import torch
from transformers import BartTokenizer
import random
from torch.utils.data import DataLoader
import config
from utils import *
import time
from model import Bart
import torch.distributed as dist
import datetime


random.seed(66)
torch.manual_seed(66)
OPTIMIZER_CONTINUE = False
torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))


def train(base_model, model_name, train_data_path, save_all=False, epochs=20, print_steps=100):
    accelerator = Accelerator()

    tokenizer = BartTokenizer.from_pretrained(config.BASE_MODEL)

    train_data = pd.read_csv(config.TCAB_DATA_PATH + "train.csv")
    val_data = pd.read_csv(config.TCAB_DATA_PATH + "val.csv")
    min_sample_num = min(50000, len(val_data))
    val_data = val_data.sample(min_sample_num)

    print("TRAIN Dataset length:{}".format(len(train_data)))
    print("VAL Dataset length:{}".format(len(val_data)))

    training_set = MyDataset(train_data, tokenizer, config.MAX_LEN)
    train_dataloader_params = {
        "batch_size": config.TRAIN_BATCH_SIZE,
        "shuffle": True,
        "num_workers": 4
    }

    valid_set = MyDataset(val_data, tokenizer, config.MAX_LEN)
    valid_dataloader_params = {
        "batch_size": config.VAL_BATCH_SIZE,
        "num_workers": 4
    }

    training_loader = DataLoader(training_set, **train_dataloader_params)
    valid_loader = DataLoader(valid_set, **valid_dataloader_params)

    model = Bart(config.BASE_MODEL)

    train_params = {
        'learning_rate': config.LEARNING_RATE
    }

    optimizer = build_optimizer("adam", model, **train_params)

    best_loss = float("inf")
    prev_model_file_path = None

    model, optimizer, training_loader, valid_loader = accelerator.prepare(
        model, optimizer, training_loader, valid_loader
    )

    for epoch in range(epochs):
        epoch = epoch
        epoch_start_time = time.time()
        losses = []
        model.train()
        for step, data in enumerate(training_loader, 0):
            optimizer.zero_grad()
            original_ids, original_mask, perturbed_ids = data["original_ids"], data["original_mask"], data[
                "perturbed_ids"]

            loss = model(original_ids, original_mask, perturbed_ids)

            if step % print_steps == 0 and step != 0:
                accelerator.print("epoch: {} , step: {} ,loss: {}".format(epoch, step, loss.item()))

            accelerator.backward(loss)
            optimizer.step()

        accelerator.wait_for_everyone()
        model = accelerator.unwrap_model(model)

        if accelerator.is_main_process:
            with torch.no_grad():
                for step, data in enumerate(valid_loader, 0):
                    model.eval()
                    original_ids, original_mask, perturbed_ids = data["original_ids"], data["original_mask"], data[
                        "perturbed_ids"]
                    valid_loss = model(original_ids, original_mask, perturbed_ids)
                    losses.append(valid_loss.item())

            loss_mean = torch.mean(torch.FloatTensor(losses))
            is_best_loss = loss_mean < best_loss
            best_loss = min(loss_mean, best_loss)

            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_loss": best_loss,
                "optimizer_state_dict": optimizer.state_dict()
            }

            prev_model_file_path = save_training_state_gpus(
                accelerator,
                train_data_path,
                base_model,
                model_name,
                epoch,
                loss_mean.item(),
                state,
                save_all,
                is_best_loss,
                prev_model_file_path
            )

            epoch_total_time = round(time.time() - epoch_start_time)
            print(" ({:d}s) - loss: {:.4f}".format(epoch_total_time, loss_mean.item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_model', type=str, default='bart-base'
    )
    parser.add_argument(
        '--model_name', type=str, default='BartModel'
    )
    parser.add_argument(
        '--train_data_path', type=str, default='train_split_None_None_None_None_success'
    )
    parser.add_argument(
        "--save_all", type=bool, default=True
    )
    parser.add_argument(
        "--epochs", type=int, default=30
    )
    parser.add_argument(
        "--print_steps", type=int, default=100
    )

    args = parser.parse_args()

    train(args.base_model,args.model_name,args.train_data_path,args.epochs,args.print_steps)