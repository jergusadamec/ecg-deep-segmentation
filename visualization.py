"""
# ECG wave segmentation

The goal is to segment different types of waves from the ECG signal. There are 4 types of waves: P-wave,
QRS-complex, T-wave, extrasystole (parts of the signal are not part of any wave). You should focus on detecting
boundaries where each wave starts and ends. You are provided with an annotated dataset of ECG signals
from 3 databases (cardiplus, incartdb, mitdb) in the data directory. There is also split.csv file containing
the paths to all the files. Each file contains one ECG signal,
also called a lead, in a JSON format. The structure of the JSON format is as follows:
```
{'data': {
        '<lead_name>': {
            'ecg': [[]],
            'label': [[]],
            'fs': int} },
 'legend': {0: 'none', 1: "p_wave", 2: "qrs", 3: "t_wave", 4: "extrasystole"}
```
The format is designed for multiple leads and multiple signals per lead, however, in this case, each file contains
exactly one lead with one signal. For each lead there is `ecg` with the ecg signal, `labels` with list
of integers [0-4] of the same length as the ecg signal (category of each point of the signal),
and `fs` which is the sampling frequency of the signal.


You are free to use the language of your choice or any other tool you would like. You are provided with two
Python functions for loading and visualization of ECG signal and the corresponding wave annotation.

You are not expected to provide the perfect solution. What counts the most is your approach and design choices.
Working pipeline with a poor performance is completely sufficient, in fact, you can choose to only describe
your approach. You should think of model/algorithm suitable for this task, any pre/post-processing, evaluation
metric. You should spend no more than 2 days on this task. For a better understanding of this task check out our blog
post: https://medical.powerful.digital/deep-learning-for-ecg-interpretation-b3ce1d094f5e
"""


import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from typing import Dict


def load_single_ecg(path: str) -> Dict[str, dict]:
    """annot format:
    {'data': {
        '<lead_name>': {
            'ecg': [[]],
            'label': [[]],
            'fs': int} },
     'legend': {0: 'none', 1: "p_wave", 2: "qrs", 3: "t_wave", 4: "extrasystole"}"""
    with open(path) as f:
        annot = json.load(f)
    return annot


def visualise_ecg(ecg: np.ndarray,
                  labels: np.ndarray,
                  pred_vec: np.ndarray = None,
                  start_offset: int = 0,
                  end_offest: int = None,
                  plot_window: int = 300,
                  max_plots: int = 2) -> None:
    """

    :param ecg: ecg signal
    :param labels: array of the same length as ecg, containing labels [0, 1, 2, 3, 4]
    :param pred_vec: same sa labels, but it is expected to be the model predictions
    :param start_offset: plot from given time-step (array index)
    :param end_offest: plot until given time-step (array index)
    :param plot_window: width of the plot window (array index units)
    :param max_plots: max number of plots to show (in case of very long ecg)
    :return:
    """
    if end_offest is None:
        end_offest = ecg.shape[1]
    y_formatter = FixedFormatter(["none", "P wave", "QRS", "T wave", "Extra\nsystole"])
    y_locator = FixedLocator([0, 1, 2, 3, 4])

    for i, start in enumerate(range(start_offset, end_offest, plot_window)):
        fig, ax1 = plt.subplots(figsize=(20, 3))
        ax1.set_ylabel('ecg', color='blue')
        ax1.plot(ecg[0, i:i + plot_window])

        ax2 = ax1.twinx()
        if pred_vec is not None:
            ax2.plot(pred_vec[start:start + plot_window], '.', color='red')
        ax2.plot(labels[0, start:start + plot_window] + 0.1, '.', color='green')
        ax2.set_ylim(ymin=-0.1, ymax=4.3)
        ax2.yaxis.set_major_formatter(y_formatter)
        ax2.yaxis.set_major_locator(y_locator)
        if i >= max_plots - 1:
            print("max_plots was reached, increase it to see more")
            break
    plt.show()


if __name__ == '__main__':
    data_root = 'resources/ecg-data'
    ecg_id = 232  # choose ecg id from [0, 289]

    # load split file containing paths to the data
    df = pd.read_csv(os.path.join(data_root, 'split.csv'))

    path = df.iloc[ecg_id]['name']
    lead_name = path.split('/')[1][:-5]  # extract lead name
    annot = load_single_ecg(os.path.join(data_root, path))

    ecg = np.array(annot['data'][lead_name]['ecg'])  # ecg signal as numpy array (for visualisation)
    label = np.array(annot['data'][lead_name]['label'])
    visualise_ecg(ecg, label, plot_window=5000)
    print("done")

