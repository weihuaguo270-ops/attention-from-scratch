"""
Speculative Decoding — 投机解码

用一个小模型草拟多个候选 token，大模型一次并行验证。
通过 rejection sampling 保证输出质量和只跑大模型完全一致。

核心直觉：
  自回归生成时，GPU 一次前向算 1 个 token 和算 4 个 token 的时间几乎一样。
  让小模型串行生成草稿，大模型批量验证 → 速度提升 2-3x，质量不变。

流程:
  Step 1: Draft model 自回归生成 γ 个候选 token（串行，但小模型快）
  Step 2: Target model 一次前向验证 γ+1 个位置（并行，大模型算）
  Step 3: Rejection sampling 逐位置判断接受/拒绝
  Step 4: 保留已接受的，从第一个拒绝的位置重新草拟

参考: https://arxiv.org/abs/2211.17192
"""
import numpy as np


def softmax(x):
    """安全的 softmax"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class SimpleLM:
    """
    简化的语言模型（用于演示 speculative decoding）

    实际中 draft 和 target 是完整 Transformer，这里用随机权重模拟。
    核心是展示 speculative decoding 的逻辑，不关注模型内部。

    参数:
        vocab_size: 词表大小
        d_model: 模型维度
        seed: 随机种子（确保可复现）
        speed_factor: 模拟速度差异（仅用于统计）
    """
    def __init__(self, vocab_size=100, d_model=32, seed=42, speed_factor=1.0):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.speed_factor = speed_factor

        rng = np.random.RandomState(seed)
        # Embedding + LM head 模拟模型输出 logits
        self.embedding = rng.randn(vocab_size, d_model) * 0.1
        self.lm_head = rng.randn(d_model, vocab_size) * 0.1

    def forward(self, token_ids):
        """
        一次前向传播，返回所有位置的 logits

        参数:
            token_ids: (seq_len,) — 输入 token 序列
        返回:
            logits: (seq_len, vocab_size) — 每个位置每个 token 的分数
        """
        # Embedding lookup
        x = self.embedding[token_ids]  # (seq_len, d_model)
        # LM head → logits
        logits = x @ self.lm_head      # (seq_len, vocab_size)
        return logits

    def generate_token(self, token_ids):
        """
        自回归生成一个 token（串行）

        参数:
            token_ids: (seq_len,) — 当前序列
        返回:
            next_token: int — 生成的 token id
            logits: (vocab_size,) — 当前位置的 logits
        """
        logits = self.forward(token_ids)
        last_logits = logits[-1]
        probs = softmax(last_logits)
        next_token = np.random.choice(self.vocab_size, p=probs)
        return next_token, last_logits

    def generate_n_tokens(self, token_ids, n):
        """
        自回归生成 n 个 token，并保存每一步的 logits

        参数:
            token_ids: (seq_len,) — 初始序列
            n: 生成多少个 token
        返回:
            tokens: (n,) — 生成的 token ids
            logits_list: [(vocab_size,), ...] — 每步的 logits
        """
        tokens = []
        logits_list = []
        current = token_ids.copy()
        for _ in range(n):
            next_token, logits = self.generate_token(current)
            tokens.append(next_token)
            logits_list.append(logits)
            current = np.append(current, next_token)
        return np.array(tokens), logits_list


class SpeculativeDecoder:
    """
    投机解码器

    用 draft model 草拟 → target model 批量验证 → rejection sampling

    参数:
        draft_model: 小模型（快，但准确率低）
        target_model: 大模型（慢，但准确率高）
        gamma: 每次草拟的候选 token 数（默认 4）
    """
    def __init__(self, draft_model, target_model, gamma=4):
        self.draft = draft_model
        self.target = target_model
        self.gamma = gamma

        # 统计信息
        self.stats = {
            "draft_calls": 0,      # 小模型前向次数
            "target_calls": 0,     # 大模型前向次数
            "accepted": 0,         # 接受的 token 数
            "rejected": 0,         # 拒绝的 token 数
            "total_tokens": 0,     # 总共生成的 token 数
        }

    @staticmethod
    def _rejection_sample(target_logits, draft_logits, draft_token):
        """
        Rejection sampling — 核心决策逻辑

        公式: 接受概率 = min(1, p_target(token) / p_draft(token))
        """
        # 目标 token 在两个模型中的概率
        p_target = softmax(target_logits)
        p_draft = softmax(draft_logits)

        p_target_tok = p_target[draft_token]
        p_draft_tok = p_draft[draft_token]

        # 接受概率
        accept_prob = min(1.0, p_target_tok / (p_draft_tok + 1e-8))

        if np.random.random() < accept_prob:
            return True, draft_token
        else:
            # 拒绝 → 从修正分布中采样
            corrected = np.maximum(0, p_target - p_draft)
            corrected_sum = corrected.sum()
            if corrected_sum > 1e-8:
                corrected = corrected / corrected_sum
                sampled = np.random.choice(len(p_target), p=corrected)
            else:
                # 修正分布无效 → 直接从 target 采样
                sampled = np.random.choice(len(p_target), p=p_target)
            return False, sampled

    def generate(self, prefix, max_new_tokens=20):
        """
        用 speculative decoding 生成文本

        参数:
            prefix: (seq_len,) — 初始 token 序列（prompt）
            max_new_tokens: 最多生成多少个新 token
        返回:
            output: (seq_len + generated,) — 完整输出序列
        """
        output = prefix.copy()

        while len(output) - len(prefix) < max_new_tokens:
            # ============================================================
            # Phase 1: Draft model 自回归生成 γ 个候选
            # ============================================================
            draft_tokens, draft_logits_list = self.draft.generate_n_tokens(
                output, min(self.gamma, max_new_tokens - (len(output) - len(prefix)))
            )
            self.stats["draft_calls"] += len(draft_tokens)

            # 构造草稿序列（prefix + draft_tokens）
            draft_seq = np.concatenate([output, draft_tokens])

            # ============================================================
            # Phase 2: Target model 一次前向验证全部位置
            # ============================================================
            target_logits_all = self.target.forward(draft_seq)
            self.stats["target_calls"] += 1

            # ============================================================
            # Phase 3: Rejection sampling 逐位置检查
            # ============================================================
            n_accepted = 0
            prefix_len = len(output)  # 固定当前序列长度

            for i in range(len(draft_tokens)):
                # 目标模型在 draft_tokens[i] 位置的 logits
                t_logits = target_logits_all[prefix_len + i]
                d_logits = draft_logits_list[i]

                accept, sampled = self._rejection_sample(
                    t_logits, d_logits, draft_tokens[i]
                )

                if accept:
                    n_accepted += 1
                    output = np.append(output, draft_tokens[i])
                    self.stats["accepted"] += 1
                else:
                    output = np.append(output, sampled)
                    self.stats["rejected"] += 1
                    # 从第一个拒绝的位置停止，重新草拟
                    break

            # 如果全部接受，额外从 target 采样一个 token
            if n_accepted == len(draft_tokens):
                last_logits = target_logits_all[-1]
                last_probs = softmax(last_logits)
                extra_token = np.random.choice(len(last_probs), p=last_probs)
                output = np.append(output, extra_token)
                self.stats["accepted"] += 1  # 虽不是草稿但算有效 token

        self.stats["total_tokens"] = len(output) - len(prefix)
        return output


# ============================================================
# 与纯自回归生成的对比
# ============================================================
def demo_speculative_decoding():
    """演示 speculative decoding 的工作流程"""
    print("=" * 60)
    print("Speculative Decoding 演示")
    print("=" * 60)

    vocab_size = 50
    np.random.seed(42)

    # 创建目标模型（大，准确）和草稿模型（小，快）
    # 这里用不同种子模拟"小模型不如大模型准"
    target = SimpleLM(vocab_size=vocab_size, d_model=32, seed=42)
    draft = SimpleLM(vocab_size=vocab_size, d_model=16, seed=99)

    decoder = SpeculativeDecoder(draft, target, gamma=4)

    # 输入 prompt（随机 token）
    prefix = np.array([5, 12, 3])

    print(f"\nPrompt tokens: {prefix}")
    print(f"目标词表大小: {vocab_size}")
    print(f"Draft 候选数 (γ): {decoder.gamma}")
    print()

    # 纯自回归生成（只用 target）
    target_only_tokens = []
    current = prefix.copy()
    for _ in range(12):
        next_tok, _ = target.generate_token(current)
        target_only_tokens.append(next_tok)
        current = np.append(current, next_tok)
    target_steps = len(target_only_tokens)

    # Speculative decoding
    output = decoder.generate(prefix, max_new_tokens=12)
    generated = output[len(prefix):]

    print(f"纯 Target 生成 {target_steps} 个 token: {target_only_tokens}")
    print(f"Spec Decoding 生成 {len(generated)} 个 token: {generated}")
    print(f"\n效率对比:")
    print(f"  Target model 前向次数（纯自回归）:  {target_steps}")
    print(f"  Target model 前向次数（Spec Decoding）: {decoder.stats['target_calls']}")
    print(f"  Speedup: {target_steps / decoder.stats['target_calls']:.2f}x")
    print(f"\nRejection sampling 统计:")
    print(f"  接受: {decoder.stats['accepted']}")
    print(f"  拒绝: {decoder.stats['rejected']}")
    if decoder.stats['accepted'] + decoder.stats['rejected'] > 0:
        print(f"  接受率: {decoder.stats['accepted'] / (decoder.stats['accepted'] + decoder.stats['rejected']):.0%}")


def demo_realistic_scenario():
    """模拟真实场景：draft 偶尔猜错，展示 rejection"""
    print("\n" + "=" * 60)
    print("真实场景模拟：Draft 偶尔猜错")
    print("=" * 60)

    vocab_size = 20
    target = SimpleLM(vocab_size=vocab_size, d_model=32, seed=42)

    # 模拟一个更"自信"的 target（logits 更尖锐 → 概率集中在少数 token）
    # 和一个更"犹豫"的 draft（logits 更平坦 → 经常猜错）
    class ConfidentLM(SimpleLM):
        def forward(self, token_ids):
            logits = super().forward(token_ids)
            return logits * 3.0  # 放大 logits → softmax 后分布更尖锐

    class HesitantLM(SimpleLM):
        def forward(self, token_ids):
            logits = super().forward(token_ids)
            return logits * 0.3  # 缩小 logits → softmax 后更均匀

    target_confident = ConfidentLM(vocab_size=vocab_size, d_model=32, seed=42)
    draft_hesitant = HesitantLM(vocab_size=vocab_size, d_model=32, seed=99)

    decoder = SpeculativeDecoder(draft_hesitant, target_confident, gamma=4)
    prefix = np.array([5, 12, 3])

    output = decoder.generate(prefix, max_new_tokens=20)

    # 纯 target 基线
    target_tokens = []
    current = prefix.copy()
    for _ in range(20):
        next_tok, _ = target_confident.generate_token(current)
        target_tokens.append(next_tok)
        current = np.append(current, next_tok)

    print(f"\n纯 Target 生成: {target_tokens[:15]}...")
    print(f"Spec Decoding:  {output[len(prefix):len(prefix)+15]}...")
    print(f"\n Target 前向（纯自回归）:  20")
    print(f" Target 前向（Spec Decoding）: {decoder.stats['target_calls']}")
    print(f" 加速: {20 / decoder.stats['target_calls']:.2f}x")
    print(f" 接受率: {decoder.stats['accepted'] / (decoder.stats['accepted'] + decoder.stats['rejected']):.0%}")
    print(f" 拒绝次数: {decoder.stats['rejected']}")
    print("\n" + "=" * 60)
    print("Gamma 值对加速比的影响分析")
    print("=" * 60)

    vocab_size = 50
    target = SimpleLM(vocab_size=vocab_size, d_model=32, seed=42)

    for gamma in [1, 2, 4, 6, 8]:
        np.random.seed(42)
        draft = SimpleLM(vocab_size=vocab_size, d_model=16, seed=99)
        decoder = SpeculativeDecoder(draft, target, gamma=gamma)
        prefix = np.array([5, 12, 3])

        # 纯自回归需要的 target 前向次数
        target_forward = 20  # 生成 20 个 token

        output = decoder.generate(prefix, max_new_tokens=20)
        speedup = target_forward / decoder.stats["target_calls"]
        accept_rate = (decoder.stats["accepted"] /
                       (decoder.stats["accepted"] + decoder.stats["rejected"] + 1e-8))

        print(f"  γ={gamma:2d} | Target前向={decoder.stats['target_calls']:2d} "
              f"| 加速={speedup:.2f}x | 接受率={accept_rate:.0%}"
              f" | 生成={len(output)-3} tokens")


if __name__ == "__main__":
    demo_speculative_decoding()
    demo_realistic_scenario()
    # demo_acceptance_rate_analysis()  # gamma 分析

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
Speculative Decoding 核心:

1. 小模型串行草拟 → 大模型并行验证
   - 小模型生成 γ 个候选 token（串行但快）
   - 大模型一次前向验证全部位置（并行，效率高）

2. Rejection Sampling 保证质量
   - 接受概率 = min(1, p_target / p_draft)
   - 拒绝后从修正分布采样
   - 最终分布 = 纯大模型采样，效果一致

3. 加速来源
   - GPU 算 1 个 token 和算 γ+1 个 token 耗时相近
   - 大模型前向次数从 O(N) 降到 O(N/γ)
   - 实际加速 2-3x（取决于接受率）
    """)
