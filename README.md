# CT-GAT
Code, data and model parameter of the EMNLP 2023 paper "**CT-GAT: Cross-Task Generative Adversarial Attack based on Transferability**"

This repository contains code modified from [Advbench](https://github.com/thunlp/Advbench), many thanks!

## Dependencies
You can create the same running environment and install dependencies as us by using the following commands:

`conda env create -f environment.yaml`

## Data Preparation and Preprocess
Please visit the `/data` folder, read the `README.md` there to obtain and process the data.

## Folder Creation
Execute the following code to create the required folders
```
mkdir param
mkdir victim
```

## Experiments
In this step, you need to operate under the `CT-GAT` directory.

First, you need to train the CT-GAT generator. You can run the following command for training. You can also directly download our parameters from Google Cloud: here. Or you can download our trained model parameters from Baidu Cloud: [here](https://pan.baidu.com/s/1U195DoiQOMpfi6FNQ4dwzw?pwd=xozm)
```
bash scripts/train_CT-GAT.sh
```


Then you should fine-tune the pre-trained model on our security datasets collection Advbench.
```
bash scripts/train_victim.sh
```

To conduct the baseline attack experiments in default settings:
```
bash scripts/base_attack.sh
```

To conduct attack experiments via ROCKET in default settings:
```
bash scripts/ROCKT.sh
```

To conduct attack experiments via CT-GAT in our settings:
```
bash scripts/CT-GAT.sh
```

## Citation
Please kindly cite our paper:
```
@misc{lv2023ctgat,
      title={CT-GAT: Cross-Task Generative Adversarial Attack based on Transferability}, 
      author={Minxuan Lv and Chengwei Dai and Kun Li and Wei Zhou and Songlin Hu},
      year={2023},
      eprint={2310.14265},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```