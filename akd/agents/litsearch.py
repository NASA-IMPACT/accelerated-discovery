import warnings

from akd.agents.search import SearchAgent

# Warning for users who have warnings enabled
warnings.warn(
    "LitAgent is deprecated. Search agents are moved to `akd.agents.search`, "
    "which has two implementations: `akd.agents.search.DeepLitSearchAgent` "
    "and `akd.agents.search.ControlledSearchAgent`",
    DeprecationWarning,
    stacklevel=2,
)


class LitAgent(SearchAgent):
    def __init__(self, *args, **kwargs):
        # Additional exception if someone tries to instantiate
        raise DeprecationWarning(
            "LitAgent is deprecated. Use `akd.agents.search.DeepLitSearchAgent` or `akd.agents.search.ControlledSearchAgent` instead.",
        )

    def _arun(self, *args, **kwargs):
        # Additional exception if someone tries to run
        raise DeprecationWarning(
            "LitAgent is deprecated. Use `akd.agents.search.DeepLitSearchAgent` or `akd.agents.search.ControlledSearchAgent` instead.",
        )
