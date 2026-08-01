"""KronosContextHead: learned ranking head on top of fine-tuned Kronos scores.

Stock selection is fully ML-driven: the Context Transformer takes
Kronos score + 20d reversal + sector momentum + CS stats + liquidity
+ market context as features and is trained with NDCG loss. The final
portfolio is the model's Top-3 (equal weight), with only risk screens
(ScreenProcessor) as constraints — no hand-crafted selection rules.
"""

from .KronosContextHead import KronosContextHeadModel

__all__ = ["KronosContextHeadModel"]
