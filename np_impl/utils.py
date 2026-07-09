"""
公共工具函数 — 被其他模块 import

import sys, os
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "np_impl"
包含 Transformer 各组件共用的基础操作：
- softmax:     Attention 中的注意力权重归一化
- split_heads: 将 Q/K/V 拆成多头，方便并行计算
- combine_heads: 将多头结果合并回原维度
- layer_norm:  对每个词做标准化，稳定训练
"""
import sys, os
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "np_impl"
import numpy as np


def softmax(x):
    """
    Softmax 归一化 — 将分数转成概率分布
    公式: softmax(x_i) = e^{x_i} / Σ e^{x_j}
    用在 Attention 的 scores → attention_weights 那一步。
    对矩阵的每一行独立做 softmax，使每行元素加起来 = 1。
    参数:
        x: shape (..., seq_len, seq_len) 的分数矩阵
    返回:
        与 x 同 shape，每行和为 1 的权重矩阵
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)
def split_heads(x, num_heads):
    """
    将 Q/K/V 拆成多个头 — 多头注意力的第一步
    输入 shape: (seq_len, d_model)
    输出 shape: (num_heads, seq_len, d_k)  其中 d_k = d_model // num_heads
    先把 (seq_len, d_model) reshape 成 (seq_len, num_heads, d_k)，
    再 transpose 成 (num_heads, seq_len, d_k)。
    这样后续矩阵乘法可以一次算出所有头的结果，不用循环。
    参数:
        x: Q/K/V 矩阵，shape (seq_len, d_model)
        num_heads: 头数
    返回:
        (num_heads, seq_len, d_k) 的拆分结果
    """
    seq_len, d_model = x.shape
    d_k = d_model // num_heads
    x = x.reshape(seq_len, num_heads, d_k)
    return x.transpose(1, 0, 2)
def combine_heads(x, num_heads):
    """
    合并多头结果 — split_heads 的逆操作
    输入 shape: (num_heads, seq_len, d_k)
    输出 shape: (seq_len, d_model)
    先把 (num_heads, seq_len, d_k) transpose 回 (seq_len, num_heads, d_k)，
    再 reshape 成 (seq_len, d_model)。
    参数:
        x: 多头注意力的输出，shape (num_heads, seq_len, d_k)
        num_heads: 头数（用于反推维度）
    返回:
        (seq_len, d_model) 的合并结果
    """
    num_heads, seq_len, d_k = x.shape
    x = x.transpose(1, 0, 2)
    return x.reshape(seq_len, -1)
def layer_norm(x, eps=1e-6):
    """
    层归一化 — 对每个词独立做标准化
    公式: LayerNorm(x) = (x - mean) / (std + eps)
    作用:
    - 让每个词的向量均值≈0，标准差≈1
    - 防止深层网络中数值过大或过小
    - 训练更稳定，收敛更快
    用在 Transformer Block 中每次残差连接之后。
    参数:
        x: 输入矩阵，shape (seq_len, d_model)
        eps: 防止除 0 的小常数
    返回:
        标准化后的矩阵，shape 与 x 相同
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)