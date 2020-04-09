import pandas as pd
import re
import matplotlib.pyplot as plt

fp = './log_training_gpu.log'

df = pd.DataFrame()
with open(fp, 'r') as f:
    txt = f.read()

    re_train_loss = re.compile('train_loss: \d.\d{10}')
    train_loss_list = re_train_loss.findall(txt)

    re_val_loss = re.compile('val_loss: \d.\d{10}')
    val_loss_list = re_val_loss.findall(txt)

    df['train_loss_txt'] = train_loss_list
    df['val_loss_txt'] = val_loss_list


def txt_to_num(x):
    return float(x.split(':')[-1])


df['train_loss'] = df['train_loss_txt'].map(txt_to_num)
df['val_loss'] = df['val_loss_txt'].map(txt_to_num)

# df['train_loss'].plot()
# df['val_loss'].plot()

# df['train_loss'].iloc[100:].plot()
df['val_loss'].iloc[100:].plot()

plt.show()
