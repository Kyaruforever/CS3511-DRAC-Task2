# CS3511-DRAC-Task2

## Environment
python 3.9

cuda >= 11.8

Pytorch correspond to cuda verion. For example, my cuda version is 12.1:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
Package install
```
pip install timm
pip install pandas
pip install scikit-learn
pip install torchnet
```

## File Structure
Before running our code, you should set up the following file structure.
```
.
├── checkpoints
│   └── modelname
├── data
│   ├── 1. Original Images
│   │   ├── a. Training Set
│   │   └── b. Testing Set
│   └── 2. Groundtruths
├── dataset.py
├── train.py
├── README.md
├── results
│   └── modelname
├── test_multi.py
└── test.py
```

`XXX.pkl` will be saved in path `./checkpoints/modelname`

`XXX.csv` will be saved in path `./result/modelname`

For example in our code, we use resnet50d as the model, the file structure:
```
.
├── checkpoints
│   └── resnet50d
├── data
│   ├── 1. Original Images
│   │   ├── a. Training Set
│   │   └── b. Testing Set
│   └── 2. Groundtruths
├── dataset.py
├── train.py
├── README.md
├── results
│   └── resnet50d
├── test_multi.py
└── test.py
```

## training
```
python main.py --model resnet50d
```
## Test
```
python test.py --model resnet50d --file-name XXX.pkl
python test_multi.py --model resnet50d
```
`test.py` is used to test one single model checkpoint.

`test_multi.py` is used to test all checkpoints of one model. 

Use `test_multi.py`, you can get a extra file `avg.csv` which is the average value of all csv file.
