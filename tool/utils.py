#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import re
import os
import shutil
import copy
import datetime
import numpy as np
import torch
import scipy
import datetime
import sys
import time

from typing import List
from collections import OrderedDict
from .logger import *

sys.setrecursionlimit(10000)


def get_specific_time():
    now = time.localtime()
    year, month, day = str(now.tm_year), str(now.tm_mon), str(now.tm_mday)
    hour, minute, second = str(now.tm_hour), str(now.tm_min), str(now.tm_sec)
    return str(year + "_" + month + "_" + day + "_" + hour + "h" + minute + "m" + second + "s")


REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def clean(x):
    x = x.lower()
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


def eval_label(match_true, pred, true, total, match):
    match_true, pred, true, match = match_true.float(), pred.float(), true.float(), match.float()
    try:
        accu = match / total
        precision = match_true / pred
        recall = match_true / true
        F = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        accu, precision, recall, F = 0.0, 0.0, 0.0, 0.0
        logger.error("[Error] float division by zero")
    return accu, precision, recall, F


# compute the cos similarity between a and b. a, b are numpy arrays
def cos_sim(self, a, b):
    return 1 - scipy.spatial.distance.cosine(a, b)


def eval_label(match_true, pred, true, total, match):
    match_true, pred, true, match = match_true.float(), pred.float(), true.float(), match.float()
    try:
        print("match_true:", match_true.data, " ;pred:", pred.data, " ;true:", true.data, " ;match:", match.data,
              " ;total:", total)
        accu = match / total
        precision = match_true / pred
        recall = match_true / true
        F = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        accu, precision, recall, F = 0.0, 0.0, 0.0, 0.0
        logger.error("[Error] float division by zero")
    return accu, precision, recall, F


def normalization(x):
    """"
    归一化到区间{0,1]
    返回副本
    """
    _range = np.max(x) - np.min(x)
    return (x - np.min(x)) / _range


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    return net
