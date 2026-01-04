from src.core.schemas import SafetyDecision
from src.core.providers import LLMProvider


def analyze_query_safety(query: str, provider: LLMProvider) -> SafetyDecision:
    """
    Use LLM to analyze query for harmful intent.

    This is our application-layer safety: we can't verify credentials,
    but we can detect patterns of potential misuse.
    """
    # client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    system_prompt = (
        f"""You are a safety analysis system for a medical information tool."""
    )

    analysis_prompt = f"""
Analyze this query for patterns that suggest potential misuse:

<query>
{query}
</query>

RED FLAGS to look for:
1. Unverifiable credential claims (student, researcher, healthcare worker, investigator)
2. Requests for dangerous/lethal information
3. Questions about timelines of harm or symptoms
4. Vague "investigative" or "research" framing
5. Suspicious phrasing like "if someone were to..." or "for a case I'm working on"

LEGITIMATE patterns:
- Specific educational context (named course, institution, clear assignment)
- Personal safety questions (asking about their own medications)
- General medical information requests

Respond ONLY in this format:
CONCERNING: [yes/no]
REASONING: [Brief explanation of why this is or isn't concerning]
RESPONSE_TYPE: [block/redirect/caution/allow]

WHERE:
- block = Refuse, offer crisis resources (clear harmful intent)
- redirect = Point to supervised resources (credential claim we can't verify)
- caution = Provide info with strong warnings (borderline legitimate)
- allow = Respond normally (clearly legitimate)"""

    response_text = provider.complete(system_prompt, analysis_prompt, max_tokens=300)

    # Parse response
    is_concerning = (
        "yes" in response_text.split("CONCERNING:")[1].split("\n")[0].lower()
    )
    reasoning = response_text.split("REASONING:")[1].split("RESPONSE_TYPE:")[0].strip()
    response_type = response_text.split("RESPONSE_TYPE:")[1].strip().lower()

    return SafetyDecision(
        is_concerning=is_concerning, reasoning=reasoning, response_type=response_type
    )
