import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from utils import packDataset_util
import torch.nn as nn
import transformers
import pandas as pd
import os
from tqdm import tqdm
# base_path = os.path.dirname(os.getcwd ())
base_path = os.path.abspath('.')
data_path = base_path +"/data/advbench/"
save_path = base_path + "/victim/"

DEVICE = 'cuda:0'

def load_data(data_name,type):
    file_path = data_path+data_name+"/"
    data = pd.read_csv(file_path+type+".csv")
    p_data = []
    for i in range(len(data)):
        p_data.append((data['text'][i], data['label'][i]))
    return p_data



def evaluaion(loader,eval_model):
    eval_model.eval()
    total_number = 0
    total_correct = 0
    with torch.no_grad():
        for padded_text, attention_masks, labels in loader:
            if torch.cuda.is_available():
                padded_text, attention_masks, labels = padded_text.to(DEVICE), attention_masks.to(DEVICE), labels.to(DEVICE)
            output = eval_model(padded_text, attention_masks).logits
            _, flag = torch.max(output, dim=1)
            total_number += labels.size(0)
            correct = (flag == labels).sum().item()
            total_correct += correct
        acc = total_correct / total_number
        return acc

import copy

def train():
    best_model = copy.deepcopy(model)
    best_acc = -1
    last_loss = 100000
    try:
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for padded_text, attention_masks, labels in tqdm(train_loader):
                if torch.cuda.is_available():
                    padded_text, attention_masks, labels = padded_text.to(DEVICE), attention_masks.to(DEVICE), labels.to(DEVICE)
                output = model(padded_text, attention_masks).logits
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            if avg_loss > last_loss:
                print('loss rise')
            last_loss = avg_loss
            print('finish training, avg_loss: {}, begin to evaluate'.format(avg_loss))
            dev_acc = evaluaion(test_loader,model)
            print('finish evaluation, acc: {}/{}'.format(dev_acc, best_acc))
            if dev_acc > best_acc:
                best_acc = dev_acc
                best_model = copy.deepcopy(model)
            print('*' * 89)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


    test_acc = evaluaion(test_loader,best_model)
    print('*' * 89)
    print('finish all, test acc: {}'.format(test_acc))
    model_path = save_path
    torch.save(best_model, model_path+dataset_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, default='satnews'
    )
    parser.add_argument(
        '--bert_type', type=str, default='bert-base-uncased'
    )
    parser.add_argument(
        '--labels', type=int, default=2
    )

    args = parser.parse_args()

    dataset_name = args.dataset
    bert_type = args.bert_type
    labels = args.labels
    EPOCHS = 8
    batch_size_dict = {'LUN':32,'satnews':32 , 'amazon_lb':64 , 'jigsaw':16 , 'EDENCE':32 , 'CGFake': 16, 'HSOL':16,'HSOL_A':16 , 'FAS':32, 'assassin':32, 'enron':32,'tweets':32,'Founta':8}



    tokenizer = AutoTokenizer.from_pretrained(bert_type)
    model = AutoModelForSequenceClassification.from_pretrained(bert_type, num_labels=labels).to(DEVICE)
    # if torch.cuda.is_available():
        # model = nn.DataParallel(model)
    model = model.to(DEVICE)
    orig_train = load_data(dataset_name,"train")
    orig_test = load_data(dataset_name,"dev")

    pack_util = packDataset_util(bert_type)
    train_loader = pack_util.get_loader(orig_train, shuffle=True, batch_size=batch_size_dict[dataset_name])  # amazon_lb 64(32)
    test_loader = pack_util.get_loader(orig_test, shuffle=False, batch_size=128)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0.1 *EPOCHS * len(train_loader), num_training_steps=EPOCHS * len(train_loader))
    train()