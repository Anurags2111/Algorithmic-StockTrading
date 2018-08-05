import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

trades = pd.read_csv('KALMANTRADES.csv')

profit = []

for var in range(0, len(trades) - 2, 2):
    print(trades['PNL'][var+1])
    profit.append(trades['PNL'][var+1] - trades['PNL'][var])

s
c = 0
for i in range(len(profit)):
    if profit[i] > 0:
        c += 1

tradeacc = (c/len(profit))*100

