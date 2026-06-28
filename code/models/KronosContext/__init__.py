"""KronosContext: Context-aware Transformer for score refinement.

Research Plan B Module 1 (B3.1): Context Transformer
- Market context features (HS300 returns, vol, breadth, dispersion)
- Industry context (sector embeddings)
- Cross-sectional statistics (CS ranks, z-scores)
- Small Transformer with [CLS] context token
"""
from .KronosContext import KronosContextModel

__all__ = ["KronosContextModel"]
