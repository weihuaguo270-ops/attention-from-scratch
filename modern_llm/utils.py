"""
Modern LLM 架构 — 公共工具函数

独立实用函数，不依赖 np_impl/。
"""
import numpy as np


def softmax(x):
    """
    Softmax 归一化
    公式: softmax(x_i) = e^{x_i} / Σ e^{x_j}
    x 可以是任意维度，在最后一维上做 softmax。
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)
