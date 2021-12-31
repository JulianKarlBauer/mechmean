#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities
"""
import numpy as np
import sys


class Ex(Exception):
    """Exception wrapping all exceptions of this package"""

    pass


def isinvertible(matrix):
    return np.linalg.cond(matrix) < 1 / sys.float_info.epsilon
