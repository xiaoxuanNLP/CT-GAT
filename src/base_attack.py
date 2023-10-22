import argparse
import torch
import numpy as np
from transformers import AutoTokenizer
import copy
import pandas as pd
from datasets import Dataset
import os
import OpenAttack
from tqdm import tqdm
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import random

random.seed(714)

DEVICE = "cuda:0"

torch.cuda.set_device(DEVICE)

# base_path = os.path.dirname(os.getcwd ())
base_path = os.path.abspath('.')
data_path = base_path + "/data/"


def load_data(data_name, type):
    file_path = data_path + data_name + "/"
    data = pd.read_csv(file_path + type + ".csv")
    p_data = []
    for i in range(len(data)):
        p_data.append((data['text'][i], data['label'][i]))
    return p_data


class MyClassifier(OpenAttack.Classifier):
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)

    # access to the classification probability scores with respect input sentences
    def get_prob(self, input_):
        inputs = self.tokenizer(input_, return_tensors='pt', return_length=512, truncation=True, padding=True)
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        if torch.cuda.is_available():
            input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        # print(inputs_ids.shape, attention_mask.shape)

        # print(self.model.device)
        outputs = self.model(input_ids, attention_mask).logits  # batch_size, labels_num
        outputs = outputs.detach().cpu().numpy()
        predicts = outputs.argmax(axis=1).tolist()
        new_outputs = []
        if dataset_name == 'sst2':
            map_li = [np.array([1, 0.]), np.array([0, 1.])]
        else:
            map_li = [np.array([1, 0., 0, 0]), np.array([0, 1., 0, 0]), np.array([0, 0., 1, 0]),
                      np.array([0, 0., 0, 1])]
        for predict_label in predicts:
            new_outputs.append(copy.deepcopy(map_li[predict_label]))
        return np.array(new_outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path', default='ag_newsbert-base-uncased'
    )
    parser.add_argument(
        '--dataset', default='ag_news'
    )
    parser.add_argument(
        '--bert_type', default='bert-base-uncased'
    )
    parser.add_argument(
        '--attacker', default='pwws'
    )
    parser.add_argument(
        '--limit_query', default=None
    )

    args = parser.parse_args()
    model_path = args.model_path
    dataset_name = args.dataset
    bert_type = args.bert_type
    attacker_name = args.attacker
    limit_query = args.limit_query

    # model = torch.load(model_path, map_location='cpu').module
    model_path = base_path + "/param/"
    model = torch.load(model_path + dataset_name, map_location=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(bert_type)

    if torch.cuda.is_available():
        model = model.to(DEVICE)

    test_data = load_data(dataset_name, "dev")
    # test_data = random.sample(test_data, 1200)

    data = []
    labels = []
    for i in range(len(test_data)):
        data.append(test_data[i][0])
        labels.append(test_data[i][1])
    test_attack = Dataset.from_dict({"x": data, "y": labels})

    victim = MyClassifier(model, tokenizer)

    # test_attack=test_attack[0:100]

    print(test_attack)
    correct_samples = [
        inst for inst in test_attack if (victim.get_pred([inst["x"]])[0] == 1 and inst["y"] == 1)
    ]

    from random import sample

    accuracy = len(correct_samples) / len(test_attack)
    print(len(correct_samples))

    if len(correct_samples) > 400:
        correct_samples = random.sample(correct_samples, 400)

    if attacker_name == "pso":
        attacker = OpenAttack.attackers.PSOAttacker()
    elif attacker_name == "textfooler":
        attacker = OpenAttack.attackers.TextFoolerAttacker()
    elif attacker_name == "pwws":
        attacker = OpenAttack.attackers.PWWSAttacker()
    elif attacker_name == "deep":
        attacker = OpenAttack.attackers.DeepWordBugAttacker()
    elif attacker_name == "bert":
        attacker = OpenAttack.attackers.BERTAttacker()
    elif attacker_name == "deep25":
        attacker = OpenAttack.attackers.DeepWordBugAttacker(power=25)
    elif attacker_name == "deep100":
        attacker = OpenAttack.attackers.DeepWordBugAttacker(power=100)
    elif attacker_name == "pso100":
        attacker = OpenAttack.attackers.PSOAttacker(max_iters=100)
    elif attacker_name == "textbugger":
        attacker = OpenAttack.attackers.TextBuggerAttacker()

    runtime = []
    querytime = []
    instances = []
    num = 0

    if limit_query is None:
        attack_eval = OpenAttack.AttackEval(
            attacker,
            victim
        )

        for result in tqdm(attack_eval.ieval(correct_samples), total=len(correct_samples)):
            runtime.append(result["metrics"]["Running Time"])
            querytime.append(result["metrics"]["Victim Model Queries"])

            # print("result['data'] = ",result['data'])
            # print("result['result'] = ",result['result'])
            # print("result['metrics'] = ",result['metrics'])
            # print("result['success'] = ",result['success'])

            if result["success"]:
                # adversarial_samples["x"].append(result["result"])
                # adversarial_samples["y"].append(result["data"]["y"])
                num = num + 1
                instances.append([result['data']["x"], result["result"], result["metrics"]["Victim Model Queries"]])

        attack_success_rate = num / len(correct_samples)

        ave_runtime = np.mean(np.array(runtime))
        ave_querytime = np.mean(np.array(querytime))
        print("ave_runtime = ", ave_runtime)
        print("ave_querytime = ", ave_querytime)

        print("------" + dataset_name + "+" + attacker_name + "------")

        name = ['origin', 'attack', 'query_time']
        save_data = pd.DataFrame(columns=name, data=instances)

        save_data.to_csv(base_path + "/output/" + dataset_name + "_" + attacker_name + '.csv')

    else:
        attack_eval = OpenAttack.AttackEval(
            attacker,
            victim,
            invoke_limit=int(limit_query)
        )

        for result in tqdm(attack_eval.ieval(correct_samples), total=len(correct_samples)):
            runtime.append(result["metrics"]["Running Time"])
            querytime.append(result["metrics"]["Victim Model Queries"])

            if result["success"]:
                num = num + 1
                instances.append([result['data']["x"], result["result"], result["metrics"]["Victim Model Queries"]])

        attack_success_rate = num / len(correct_samples)

        ave_runtime = np.mean(np.array(runtime))
        ave_querytime = np.mean(np.array(querytime))
        print("ave_runtime = ", ave_runtime)
        print("ave_querytime = ", ave_querytime)
        print("attack_success_rate = ", attack_success_rate)

        print("------" + dataset_name + "+" + attacker_name + "------")

        name = ['origin', 'attack', 'query_time']
        save_data = pd.DataFrame(columns=name, data=instances)

        save_data.to_csv(base_path + "/output/" + dataset_name + "_" + attacker_name + '.csv')


