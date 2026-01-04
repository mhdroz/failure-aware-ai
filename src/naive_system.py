from src.core.schemas import DrugPair
from src.core.providers import LLMProvider

NAIVE_SYSTEM_PROMPT = """You are a helpful assistant that provides information about drug interactions.

When given two medications, assess whether they interact and provide guidance.

Keep your response brief and helpful."""


def naive_interaction_check(
    drug_pair: DrugPair, provider: LLMProvider, max_tokens: int = 300
) -> str:
    """
    System 1: Naive implementation
    - Single LLM call
    - No database checking
    - Returns whatever the model says
    """

    user_prompt = f"""Do {drug_pair.drug1} and {drug_pair.drug2} interact?"""

    if drug_pair.context:
        user_prompt += f"\n\nContext: {drug_pair.context}"

    response_text = provider.complete(
        NAIVE_SYSTEM_PROMPT, user_prompt, max_tokens=max_tokens
    )

    return response_text
