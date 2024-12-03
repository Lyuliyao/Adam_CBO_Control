Set Variable dim = 2 4 8 16 32 64 respectively
```
python3 -u train.py --dim ${dim}
python3 -u simulation.py ${dim}
python3 -u result.py ${dim}
```