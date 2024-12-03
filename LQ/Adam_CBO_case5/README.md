```
conda activate CBO
python3 -u train.py --batch_size 64
python3 -u train.py --batch_size 128
python3 -u train.py --batch_size 256
python3 -u train.py --batch_size 512
python3 -u train.py --batch_size 1024
python3 -u train.py --batch_size 2048
python3 -u result.py --batch_size 64
python3 -u result.py --batch_size 128
python3 -u result.py --batch_size 256
python3 -u result.py --batch_size 512
python3 -u result.py --batch_size 1024
python3 -u result.py --batch_size 2048
```