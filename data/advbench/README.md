# Data Preparation
1. You need to download the data.zip dataset from Data Preparation in `Advbench` ([https://github.com/thunlp/Advbench](https://github.com/thunlp/Advbench/tree/main#data-preparation)) and unzip this package to the current directory (a total of 10 datasets). Please note, you do not need to create a new data folder in this path.
2. You need to obtain the `Founta` dataset ([https://github.com/ENCASEH2020/hatespeech-twitter](https://github.com/ENCASEH2020/hatespeech-twitter)) named `hatespeech_text_label_vote_RESTRICTED_100K.csv` and place it in this folder, then execute the following command:

```
mkdir Founta
mv hatespeech_text_label_vote_RESTRICTED_100K.csv Founta
python preprocess.py
```