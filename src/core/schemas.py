from dataclasses import dataclass


@dataclass
class DrugPair:
    drug1: str
    drug2: str
    context: str = ""


@dataclass
class SafetyDecision:
    is_concerning: bool
    reasoning: str
    response_type: str  # 'block', 'redirect', 'caution', 'allow'
