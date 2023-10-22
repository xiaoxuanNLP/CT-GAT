import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import argparse
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import pandas as pd
import random
random.seed(714)
import os
from model import Bart
# base_path = os.path.dirname(os.getcwd ())
base_path = os.path.abspath('.')
data_path = base_path +"/data/advbench/"
BART_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
import config
import re

def load_data(data_name, type):
    file_path = data_path + data_name + "/"
    print("dataset = ", file_path + type + ".csv")
    data = pd.read_csv(file_path + type + ".csv")
    p_data = []
    for i in range(len(data)):
        p_data.append((data['text'][i], data['label'][i]))
    return p_data

def get_lowest_loss_file(directory):
    min_loss = float('inf')
    min_loss_file = None

    for filename in os.listdir(directory):
        match = re.search(r'loss_(\d+\.\d+)', filename)
        if match:
            loss = float(match.group(1))
            if loss < min_loss:
                min_loss = loss
                min_loss_file = filename

    return directory + min_loss_file

def get_output_label(sentence, tokenizer, model):
    if pd.isna(sentence):
        return -1
    model = model.to(DEVICE)
    tokenized_sent = tokenizer(sentence, return_tensors='pt', max_length=512, truncation=True)
    input_ids, attention_mask = tokenized_sent['input_ids'].to(DEVICE), tokenized_sent['attention_mask'].to(DEVICE)
    output_label = model(input_ids, attention_mask).logits.squeeze().argmax()
    return output_label


def correct_print(asn, total):
    print("Attack Success")
    print('ASR: ', asn / total)


def is_chinese(ch):
    if ch < '一' or ch > '龥':
        return False
    return True


def content_chinese(sentence):
    for word in sentence:
        if is_chinese(word) is True:
            return True
    return False


def attack(model, orig_test, tokenizer, ATTACK_ITER, task_name, tem):
    # test
    bart_model = Bart("/mnt/lmx_home/AttackTextGenerator/base/bart-base", device=BART_DEVICE)
    # bart_model = Bart(config.BASE_MODEL, device=BART_DEVICE)
    param_file = get_lowest_loss_file(base_path+"/param/")
    checkpoint = torch.load(param_file)

    bart_model.load_state_dict(checkpoint['model_state_dict'])
    bart_model.to(BART_DEVICE)

    correct = 0
    correct_samples = []
    for sentence, label in tqdm(orig_test):
        output_label = get_output_label(sentence, tokenizer, model)
        if output_label == label:
            correct += 1.
            if label == 1:
                correct_samples.append((sentence, 1))

    print('Orig Acc: ', correct / len(orig_test))

    if len(correct_samples) > 500:
        correct_samples = random.sample(correct_samples, 500)

    asn = 0
    total = 0
    success_list = []
    querytime = 0
    print(ATTACK_ITER)

    for sentence, label in tqdm(correct_samples):
        if content_chinese(sentence):
            continue
        querytime_each = 0
        total = total + 1
        sentence = sentence.lower()
        orig_sentence = sentence
        orig_output_label = get_output_label(sentence, tokenizer, model)
        if orig_output_label == label:

            sentences = bart_model.generate(sentence, return_num=ATTACK_ITER,tem=tem)
            for sentence in sentences:
                output_label = get_output_label(sentence, tokenizer, model)
                querytime = querytime + 1
                querytime_each = querytime_each + 1
                if output_label != label:
                    asn += 1
                    success_list.append([orig_sentence,sentence,querytime_each])
                    break
        else:
            continue

    print(querytime)
    print(asn)
    name = ['origin','attack',"querytime"]
    save_data = pd.DataFrame(columns=name,data=success_list)
    save_data.to_csv(base_path+"/output/"+task_name+'_CT-GAT.csv')

    print('ASR: ', asn / total)
    print("query: ", querytime / total)


def pipe(dataset_name, attack_iter, tem):
    bert_type = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(bert_type)
    model_path = base_path + "/victim/"
    print("start")
    model = torch.load(model_path + dataset_name, map_location=DEVICE)

    model = model.to(DEVICE)
    print("load victim finished")

    orig_test = load_data(dataset_name, "dev")

    ATTACK_ITER = attack_iter
    attack(model, orig_test, tokenizer, ATTACK_ITER, dataset_name, tem)
    print("-" * 10 + dataset_name + "+CT-GAT" + "+" + str(ATTACK_ITER) + "tem." + str(tem) + "-" * 10)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='amazon_lb', type=str)
    parser.add_argument(
        '--limit_query', default=50, type=int
    )
    parser.add_argument(
        '--tem', type=float, default=2.1
    )
    args = parser.parse_args()
    print("test")

    pipe(args.name,args.limit_query,args.tem)

