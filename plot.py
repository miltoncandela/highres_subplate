import pandas as pd
from pickle import load
import matplotlib.pyplot as plt

model_name = 'B'
x = load(open(model_name+'/history/cor.pkl', 'rb'))

e = range(1, len(x['loss']) + 1)
fig = plt.figure()
plt.plot(e, x['loss'], label = 'Training', color = 'blue')
plt.plot(e, x['val_loss'], label = 'Validation', color = 'orange')
plt.ylabel('Dice loss')
plt.xlabel('Epoch')
plt.legend()
fig.savefig(model_name+'A/cor_loss.png')

fig = plt.figure()
plt.plot(e, x['dice_coef'], label = 'Training', color = 'blue')
plt.plot(e, x['val_dice_coef'], label = 'Validation', color = 'orange')
plt.ylabel('Dice coefficient')
plt.xlabel('Epoch')
plt.legend()
fig.savefig(model_name+'/cor_dice.png')
