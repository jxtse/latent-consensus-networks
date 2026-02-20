"""Latent Consensus Networks (LCN) - Hierarchical Multi-Agent Collaboration"""

__version__ = "0.1.0"

# Imports will be added as modules are implemented
__all__ = [
    "LCNAgent",
    "HierarchicalKVCache",
    "CrossLevelAttention",
    "ConsensusProtocol",
]

def __getattr__(name):
    """Lazy imports to allow incremental development."""
    if name == "LCNAgent":
        from lcn.core.agent import LCNAgent
        return LCNAgent
    elif name == "HierarchicalKVCache":
        from lcn.core.kv_cache import HierarchicalKVCache
        return HierarchicalKVCache
    elif name == "CrossLevelAttention":
        from lcn.core.attention import CrossLevelAttention
        return CrossLevelAttention
    elif name == "ConsensusProtocol":
        from lcn.core.consensus import ConsensusProtocol
        return ConsensusProtocol
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
