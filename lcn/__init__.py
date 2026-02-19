"""Latent Consensus Networks (LCN) - Hierarchical Multi-Agent Collaboration"""

__version__ = "0.1.0"

from lcn.core.agent import LCNAgent
from lcn.core.kv_cache import HierarchicalKVCache
from lcn.core.attention import CrossLevelAttention
from lcn.core.consensus import ConsensusProtocol

__all__ = [
    "LCNAgent",
    "HierarchicalKVCache",
    "CrossLevelAttention",
    "ConsensusProtocol",
]
