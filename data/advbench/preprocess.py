import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

with open("./Founta/hatespeech_text_label_vote_RESTRICTED_100K.csv",'r',encoding='utf-8') as f:
    lines = f.readlines()

lines = lines[3:] # There is something wrong with the format of the first three lines and needs to be deleted.
lines.insert(0, 'text\tLabel\tVotes\n')
with open('./Founta/Founta.csv', 'w',encoding='utf-8') as f:
    f.writelines(lines)

df = pd.read_csv("./Founta/Founta.csv",sep="\t",error_bad_lines=False,engine='python')
df = df[df['Label'].isin(['abusive', 'hateful', 'normal'])]
df['Label'] = df['Label'].map({'abusive': 1, 'hateful': 1, 'normal': 0})

df = df.rename(columns={'Label': 'label'})

df = df.drop(columns=['Votes'])

df = shuffle(df)

train, dev = train_test_split(df, test_size=0.2)
train.to_csv('./Founta/train.csv', index=False)
dev.to_csv('./Founta/dev.csv', index=False)


