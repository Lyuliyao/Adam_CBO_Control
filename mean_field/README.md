# code for the mean field control probelm.

To solve the control problem,

```
python3 -u train.py --dim 1
```

To test the control model under different circumstance, one can run
```
python3 -u result_1.py --N_mv 50
python3 -u result_2.py --N_mv 50
python3 -u result_3.py --N_mv 50
```
Here *N_mv* controls the number of agents and different result set the distribution of the agents.