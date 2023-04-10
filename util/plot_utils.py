# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path, PurePath
import math
import numpy as np


def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt', num_epoch=1):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    '''
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(
                f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # verify valid dir(s) and that every item in list is Path object
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(
                f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if dir.exists():
            continue
        raise ValueError(
            f"{func_name} - invalid directory in logs argument:\n{dir}")

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True,
                        nrows=num_epoch) for p in logs]
    ncols = int(round(math.sqrt(len(fields))))
    nrows = int(math.ceil(len(fields) / ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(5 * ncols, 5 * nrows))
    colors = sns.color_palette(n_colors=len(logs))
    for df, color in zip(dfs, colors):
        for j, field in enumerate(fields):
            # print(j, j // ncols, j % ncols)
            if field == 'mAP':
                coco_eval = pd.DataFrame(pd.np.stack(df.test_coco_eval_bbox.dropna().values)[
                    :, 1]).ewm(com=ewm_col).mean()
                # print(axs)
                axs.plot(coco_eval, c=color)
                return fig, axs

            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axs[j // ncols][j % ncols],
                    color=[color] * 2,
                    style=['-', '--']
                )

    # edit each ax
    for i, field in enumerate(fields):
        if len(fields) == 1:
            ax = axs
        else:
            ax = axs[i // ncols][i % ncols]
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(field)
        if field in "mAP":
            return fig, axs
        ax.get_legend().remove()
    # legend
    lines = axs[0][0].get_lines()
    legend_line = fig.legend([lines[i] for i in range(len(logs))], [
        "train", "test"], loc=8, bbox_to_anchor=(0.6, 0.01))
    legend_color = fig.legend([lines[2 * i] for i in range(len(logs))], [Path(
        p).name for p in logs], loc=8, bbox_to_anchor=(0.4, 0.01))
    fig.add_artist(legend_line)
    fig.tight_layout()
    fig.subplots_adjust(bottom=1.0 / (5 * nrows))
    return fig, axs

def plot_mAP(logs, ewm_col=0, log_name='log.txt', num_epoch=1):
    dfs = pd.read_json(Path(logs) / log_name, lines=True, nrows=num_epoch)
    # ncols = int(round(math.sqrt(len(fields))))
    # nrows = int(math.ceil(len(fields) / ncols))

    dfs  = dfs["test_coco_eval_bbox"]
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.251
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.444
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.254
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.109
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.283
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.360
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.241
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.403
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.435
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.192
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.483
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.639
    print(type(dfs[0]))

def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        print(f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' +
              f'score={scores.mean():0.3f}, ' +
              f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
              )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title('Precision / Recall')
    axs[0].legend(names)
    axs[1].set_title('Scores / Recall')
    axs[1].legend(names)
    return fig, axs
