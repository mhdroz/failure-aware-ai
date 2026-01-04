from src.core.providers import LLMProvider
from src.safety_layer import analyze_query_safety


def call_medical_llm(
    query: str,
    provider: LLMProvider,
    safety_reasoning: str = None,
    response_type: str = None,
) -> str:
    """
    Call medical LLM with the safety concerns as context.
    """

    # Use the safety reasoning to guide the response
    system_prompt = f"""You are a helpful medical information assistant.
    You provide accurate, evidence-based information about medications, drug interactions, 
and medical topics for educational purposes. 

Always remind users that this is educational information and they should consult 
healthcare professionals for personal medical decisions."""

    if safety_reasoning is not None and response_type is not None:

        prompt = f"""
    {query}

    WARNING: This query was flagged with the following concern:
    {safety_reasoning}

    The safety system has determined that the appropriate response type is: 
    {response_type}

    Based on this analysis, provide a response that:
    - Follows the guidelines for the specified response type
    - Addresses the general educational question
    - Provides only high-level principles and categories
    - Does NOT provide specific dangerous details
    - Emphasizes supervised learning resources
    - Maintains an educational but cautious tone

    DO NOT provide detailed instructions, doses, timelines, or specific harmful information.
    """
    else:
        prompt = query

    response_text = provider.complete(system_prompt, prompt, max_tokens=3000)

    return response_text


def safe_medical_assistant(query: str, provider: LLMProvider) -> str:
    """
    Medical assistant with intent-aware safety layer.
    """
    # Analyze query safety
    safety = analyze_query_safety(query, provider)

    # If blocked - return the safety analysis reasoning
    # It already explains the concern and suggests appropriate resources
    if "block" in safety.response_type:
        return {"safety_analysis": safety, "response": safety.reasoning}

    # For caution or redirect - call medical LLM with the safety reasoning and response_type as context
    if "caution" in safety.response_type or "redirect" in safety.response_type:

        return {
            "safety_analysis": safety,
            "response": call_medical_llm(
                query=query,
                provider=provider,
                safety_reasoning=safety.reasoning,
                response_type=safety.response_type,
            ),
        }

    # For allow - normal response
    return {
        "safety_analysis": safety,
        "response": call_medical_llm(query=query, provider=provider),
    }
