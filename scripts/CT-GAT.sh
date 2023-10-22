#!bash
dataset_name=('amazon_lb' 'CGFake' 'Founta' 'jigsaw' 'EDENCE' 'FAS' 'LUN' 'satnews')

for i in {0..7}
do
    python src/CT-GAT.py --name ${dataset_name[i]} --limit_query 100
done

python src/CT-GAT.py --name assassin --limit_query 60
python src/CT-GAT.py --name enron --limit_query 40