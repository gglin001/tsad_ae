import matplotlib.pyplot as plt
import numpy as np


def norm_func_ibi(arr, max_rri=2000, min_rri=300):
    out = (arr - min_rri) / (max_rri - min_rri)
    return out


def norm_func_hr(arr, max_hr=200, min_hr=30):
    out = (arr - min_hr) / (max_hr - min_hr)
    return out


def convert_normed_rri(normed_rri, max_rri=2000, min_rri=300):
    rri = min_rri + normed_rri * (max_rri - min_rri)
    hr = 6e4 / rri
    return rri, hr


def view_res(y_t, y_p, data_x, label_flag=True, show_all=False):
    if label_flag:
        y_t = np.asarray([np.argmax(x) for x in y_t])
        y_p = np.asarray([np.argmax(x) for x in y_p])

    minus = y_p - y_t
    minus_abs = abs(minus)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
    axs = axs.ravel()
    nums, bins, _ = axs[0].hist(minus, bins=30, ec='m', label='minus')
    for n, b in zip(nums, bins):
        if n != 0.:
            axs[0].text(x=b, y=n, s=f'{n:.0f}')
    nums, bins, _ = axs[1].hist(minus_abs, bins=30, ec='m', label='minus abs')
    for n, b in zip(nums, bins):
        if n != 0.:
            axs[1].text(x=b, y=n, s=f'{n:.0f}')
    for ax in axs:
        ax.legend()
    plt.show()

    if not show_all:
        error_res_idxs = np.where(minus_abs > 0.5)[0]
        print((f'error_res_idxs: {len(error_res_idxs)}, total_idxs: {len(y_t)}',
               f'error_ratio: {len(error_res_idxs)/len(y_t) * 100:.4f}%',
               f'Acc: {(1-len(error_res_idxs)/len(y_t)) * 100:.4f}%'))
    else:
        error_res_idxs = np.arange(len(minus_abs))

    current_idx = 0
    while(current_idx < len(error_res_idxs)):
        fg, axs = plt.subplots(4, 5, num=3, figsize=(16, 8))
        axs = axs.ravel()
        bxs = [x.twinx() for x in axs]

        for idx in range(20):
            if not current_idx + idx < len(error_res_idxs):
                break
            idx_in_raw_data = error_res_idxs[current_idx + idx]
            rri_normed = data_x[idx_in_raw_data].ravel()
            rri, hr = convert_normed_rri(rri_normed)

            axs[idx].plot(rri, '-b.', label='rri')
            axs[idx].set_ylim(300, 2000)
            bxs[idx].plot(hr, '-r.', label='hr')
            bxs[idx].set_ylim(30, 200)

            axs[idx].set_title(f'raw: {y_t[idx_in_raw_data]}, predict: {y_p[idx_in_raw_data]}')
            axs[idx].legend(loc=2) if idx == 0 else None
            bxs[idx].legend(loc=1) if idx == 0 else None
        plt.suptitle('predict failed ibis')
        plt.tight_layout()
        plt.show()

        current_idx += idx
