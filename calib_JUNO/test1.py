#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import time
import sys


def print11(*args, **kwargs):
    kwargs{'flush'} = True
    print(*args, **kwargs)

for i in np.arange(1e8):
    time.sleep(2)
    print11(i)
