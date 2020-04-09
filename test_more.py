import numpy as np
import pandas as pd

from import_model import *
from apply_model import apply_model
from utils import norm_func_ibi, view_res


def apply_anormal_raw():
    fp = 'rew_anormal.pickle'
    df = pd.read_pickle(fp)
    print(fp)
    print(df.shape)

    df.columns = ['rri']
    df['rri_norm_lim'] = df['rri'].map(norm_func_ibi)
    rri = np.vstack(df['rri_norm_lim'].values)
    rri = rri.reshape((len(rri), 1, -1))

    y_p = apply_model(rri)
    y_t = np.ones_like(y_p)

    view_res(y_t, y_p, rri, label_flag=False)


if __name__ == "__main__":
    apply_anormal_raw()
