"""Granite Guardian risk categories."""

from akd.guardrails.categories._base import RiskCategory, RiskMetadata


class GraniteRiskCategory(RiskCategory):
    """
    Input risk categories for Granite Guardian single-risk mode (snake_case values).

    Used as INPUT to specify which risk to check with granite3-guardian 2B/8B models.
    """

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


class GraniteHarmCategory(RiskCategory):
    """
    Output harm categories from Granite Guardian multi-harm model (Title Case values).

    Returned as OUTPUT from the granite-guardian-3.2-5b-multi-harm-GGUF model.
    The model returns comma-separated Title Case category names.

    The `extra.maps_to` field contains the corresponding GraniteRiskCategory value
    for mapping between input/output formats.
    """

    # Harm categories (Title Case as returned by model)
    SOCIAL_BIAS = (
        "Social Bias",
        RiskMetadata(
            description="Socially biased content",
            extra={"maps_to": "social_bias"},
        ),
    )
    JAILBREAKING = (
        "Jailbreaking",
        RiskMetadata(
            description="Jailbreak attempts",
            severity="high",
            extra={"maps_to": "jailbreak"},
        ),
    )
    VIOLENCE = (
        "Violence",
        RiskMetadata(
            description="Violence-related content",
            severity="high",
            extra={"maps_to": "violence"},
        ),
    )
    PROFANITY = (
        "Profanity",
        RiskMetadata(
            description="Profane language",
            extra={"maps_to": "profanity"},
        ),
    )
    SEXUAL_CONTENT = (
        "Sexual Content",
        RiskMetadata(
            description="Sexual content",
            severity="high",
            extra={"maps_to": "sexual_content"},
        ),
    )
    UNETHICAL_BEHAVIOR = (
        "Unethical Behavior",
        RiskMetadata(
            description="Unethical behavior",
            extra={"maps_to": "unethical_behavior"},
        ),
    )
    HARMFUL = (
        "Harmful",
        RiskMetadata(
            description="General harmful content",
            severity="high",
            extra={"maps_to": "harm"},
        ),
    )

    # Non-harmful markers (model returns these when content is safe)
    NOT_HARMFUL_PROMPT = (
        "Not harmful prompt",
        RiskMetadata(
            description="Prompt is not harmful",
            severity="low",
        ),
    )
    NOT_HARMFUL_RESPONSE = (
        "Not harmful response",
        RiskMetadata(
            description="Response is not harmful",
            severity="low",
        ),
    )
