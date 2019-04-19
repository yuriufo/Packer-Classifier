#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import torch
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix


class RunningAverageMeter(object):
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)


def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def update_train_state(model, train_state):
    # 详细信息
    print(
        "[EPOCH]: {0} | [LR]: {1} | [TRAIN LOSS]: {2:.2f} | [TRAIN ACC]: {3:.1f}% | [VAL LOSS]: {4:.2f} | [VAL ACC]: {5:.1f}% | [NFE-F]: {6:.1f} | [NFE-B]: {7:.1f}"
        .format(train_state['epoch_index'], train_state['learning_rate'],
                train_state['train_loss'][-1], train_state['train_acc'][-1],
                train_state['val_loss'][-1], train_state['val_acc'][-1],
                train_state['f_nfe'][-1], train_state['b_nfe'][-1]))

    # 至少保存一次模型
    if train_state['epoch_index'] == 0:
        torch.save(
            model.state_dict(),
            str(train_state['save_dir'] / train_state['model_filename']))
        train_state['stop_early'] = False
    # 如果模型性能表现有提升，再次保存
    else:
        loss_tm1, loss_t = train_state['val_loss'][-2:]

        # 如果损失增大
        if loss_t >= train_state['early_stopping_best_val']:
            # 更新步数
            train_state['early_stopping_step'] += 1
        # 损失变小
        else:
            # 保存最优的模型
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(
                    model.state_dict(),
                    str(train_state['save_dir'] /
                        train_state['model_filename']))

            # 重置早停的步数
            train_state['early_stopping_step'] = 0
            train_state['early_stopping_best_val'] = loss_t

        # 是否需要早停
        train_state['stop_early'] = train_state[
            'early_stopping_step'] >= train_state['early_stopping_criteria']
    return train_state


def img_to_array(fp):
    img = Image.open(fp)
    array = np.asarray(img, dtype="float32")
    return array


def save_train_state(train_state, save_dir):
    train_state["done_training"] = True
    train_state['save_dir'] = str(train_state['save_dir'])
    with save_dir.open("w") as fp:
        json.dump(train_state, fp)
    print("---->>>  Training complete!")


def plot_performance(train_state, save_dir, show_plot=True):
    """ Plot loss and accuracy.
    """
    # Figure size
    plt.figure(figsize=(15, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.plot(train_state["train_loss"], label="train")
    plt.plot(train_state["val_loss"], label="val")
    plt.legend(loc='upper right')

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plt.plot(train_state["train_acc"], label="train")
    plt.plot(train_state["val_acc"], label="val")
    plt.legend(loc='lower right')

    # Save figure
    plt.savefig(str(save_dir))

    # Show plots
    if show_plot:
        # print("---->>>   Metric plots:")
        plt.show()


def Confusion_matrix(y_pred, y_target):
    cm = confusion_matrix([i.cpu().numpy().tolist() for i in y_target],
                          [i.cpu().numpy().tolist() for i in y_pred])
    print(cm)
