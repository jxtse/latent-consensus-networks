# lcn/core/results.py
"""Result dataclasses for consensus formation."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ConsensusResult:
    """
    Result of a consensus formation process.

    Captures the outcome of a multi-round consensus formation run,
    including the final decision, per-agent decisions, attention history,
    and convergence information.

    Attributes:
        decision: Final group decision (e.g., "Candidate C")
        agent_decisions: Per-agent decisions mapping agent_id to decision string
        attention_history: List of attention weight dicts over consensus rounds
        convergence_round: Round when consensus was reached, or None if not reached
    """

    decision: str
    agent_decisions: Dict[int, str]
    attention_history: List[Dict]
    convergence_round: Optional[int]
