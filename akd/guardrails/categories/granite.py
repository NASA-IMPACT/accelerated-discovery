"""Granite Guardian risk categories."""

from akd.guardrails.categories._base import RiskCategory, RiskMetadata


class GraniteRiskCategory(RiskCategory):
    """Risk categories for Granite Guardian models (snake_case values)."""

    # Core harm categories - tuple: (value, RiskMetadata)
    HARM = ("harm", RiskMetadata(description="General harmful content", severity="high"))
    SOCIAL_BIAS = ("social_bias", RiskMetadata(description="Socially biased content"))
    PROFANITY = ("profanity", RiskMetadata(description="Profane language"))
    SEXUAL_CONTENT = ("sexual_content", RiskMetadata(description="Sexual content", severity="high"))
    UNETHICAL_BEHAVIOR = ("unethical_behavior", RiskMetadata(description="Unethical behavior"))
    VIOLENCE = ("violence", RiskMetadata(description="Violence-related content", severity="high"))
    JAILBREAK = ("jailbreak", RiskMetadata(description="Jailbreak attempts", severity="high"))

    # RAG categories
    GROUNDEDNESS = ("groundedness", RiskMetadata(description="Response grounded in context"))
    RELEVANCE = ("relevance", RiskMetadata(description="Content relevance to query"))
    ANSWER_RELEVANCE = ("answer_relevance", RiskMetadata(description="Answer relevance to question"))
