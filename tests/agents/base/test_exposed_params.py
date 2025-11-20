"""Test cases for exposed parameter functionality."""

from akd._base.utils import exposed_param

from .conftest import TestInstructorBaseAgent


class TestAgentWithExposedParams(TestInstructorBaseAgent):
    """Test agent with exposed parameters."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self._clarification_prompt = "Default clarification prompt"
        self._max_iterations = 5
        self._temperature_override = 0.7

    @exposed_param(description="System prompt for query clarification")
    def clarification_prompt(self) -> str:
        return self._clarification_prompt

    @clarification_prompt.setter
    def clarification_prompt(self, val: str) -> None:
        self._clarification_prompt = val

    @exposed_param(description="Maximum research iterations", min=1, max=20)
    def max_iterations(self) -> int:
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, val: int) -> None:
        self._max_iterations = val

    @exposed_param  # No description - uses docstring or property name
    def temperature_override(self) -> float:
        """Temperature override for this agent"""
        return self._temperature_override

    @temperature_override.setter
    def temperature_override(self, val: float) -> None:
        self._temperature_override = val

    @exposed_param(description="Read-only computed value")
    def computed_value(self) -> str:
        """This property has no setter, so it's read-only."""
        return f"iterations={self._max_iterations}, temp={self._temperature_override}"


# Prevent pytest from collecting this as a test
TestAgentWithExposedParams.__test__ = False


class TestExposedParamFunctionality:
    """Test exposed parameter functionality."""

    def test_get_exposed_params_without_values(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test get_exposed_params returns metadata without values."""
        agent = TestAgentWithExposedParams()

        exposed = agent.get_exposed_params(include_values=False)

        # Verify all exposed params are present
        assert "clarification_prompt" in exposed
        assert "max_iterations" in exposed
        assert "temperature_override" in exposed
        assert "computed_value" in exposed

        # Verify metadata structure
        assert exposed["clarification_prompt"]["description"] == "System prompt for query clarification"
        assert exposed["clarification_prompt"]["type"] == "str"
        assert exposed["clarification_prompt"]["editable"] is True

        # Verify extra metadata
        assert "extra" in exposed["max_iterations"]
        assert exposed["max_iterations"]["extra"]["min"] == 1
        assert exposed["max_iterations"]["extra"]["max"] == 20

        # Verify no values included
        assert "current_value" not in exposed["clarification_prompt"]

    def test_get_exposed_params_with_values(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test get_exposed_params returns metadata with current values."""
        agent = TestAgentWithExposedParams()

        exposed = agent.get_exposed_params(include_values=True)

        # Verify values are included
        assert exposed["clarification_prompt"]["current_value"] == "Default clarification prompt"
        assert exposed["max_iterations"]["current_value"] == 5
        assert exposed["temperature_override"]["current_value"] == 0.7

    def test_exposed_param_auto_description_from_docstring(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test that exposed_param uses docstring when no description provided."""
        agent = TestAgentWithExposedParams()

        exposed = agent.get_exposed_params()

        # temperature_override has no explicit description, should use docstring
        assert exposed["temperature_override"]["description"] == "Temperature override for this agent"

    def test_exposed_param_editable_detection(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test that editable flag is auto-detected from setter presence."""
        agent = TestAgentWithExposedParams()

        exposed = agent.get_exposed_params()

        # Parameters with setters should be editable
        assert exposed["clarification_prompt"]["editable"] is True
        assert exposed["max_iterations"]["editable"] is True
        assert exposed["temperature_override"]["editable"] is True

        # Parameter without setter should be read-only
        assert exposed["computed_value"]["editable"] is False

    def test_exposed_param_setter_functionality(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test that setters work correctly for exposed params."""
        agent = TestAgentWithExposedParams()

        # Test initial values
        assert agent.clarification_prompt == "Default clarification prompt"
        assert agent.max_iterations == 5

        # Test setting new values
        agent.clarification_prompt = "New prompt"
        agent.max_iterations = 10

        # Verify values changed
        assert agent.clarification_prompt == "New prompt"
        assert agent.max_iterations == 10

        # Verify reflected in get_exposed_params
        exposed = agent.get_exposed_params(include_values=True)
        assert exposed["clarification_prompt"]["current_value"] == "New prompt"
        assert exposed["max_iterations"]["current_value"] == 10

    def test_exposed_param_extra_metadata(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test that extra metadata is properly stored and retrieved."""
        agent = TestAgentWithExposedParams()

        exposed = agent.get_exposed_params()

        # max_iterations has extra metadata (min, max)
        assert "extra" in exposed["max_iterations"]
        assert exposed["max_iterations"]["extra"]["min"] == 1
        assert exposed["max_iterations"]["extra"]["max"] == 20

        # clarification_prompt has no extra metadata
        assert "extra" not in exposed["clarification_prompt"]

    def test_exposed_param_type_inference(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test that types are correctly inferred from current values."""
        agent = TestAgentWithExposedParams()

        exposed = agent.get_exposed_params()

        assert exposed["clarification_prompt"]["type"] == "str"
        assert exposed["max_iterations"]["type"] == "int"
        assert exposed["temperature_override"]["type"] == "float"
        assert exposed["computed_value"]["type"] == "str"

    def test_regular_properties_not_exposed(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test that regular @property decorated attributes are not exposed."""
        agent = TestAgentWithExposedParams()

        exposed = agent.get_exposed_params()

        # These are regular properties from BaseAgent, should NOT be exposed
        assert "memory" not in exposed
        assert "description" not in exposed
        assert "client" not in exposed

    def test_exposed_param_with_complex_types(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test exposed params with complex return types."""

        class ComplexAgent(TestInstructorBaseAgent):
            _config_dict = {"key": "value"}

            @exposed_param(description="Configuration dictionary")
            def config_dict(self) -> dict:
                return self._config_dict

            @config_dict.setter
            def config_dict(self, val: dict) -> None:
                self._config_dict = val

        ComplexAgent.__test__ = False

        agent = ComplexAgent()
        exposed = agent.get_exposed_params(include_values=True)

        assert exposed["config_dict"]["type"] == "dict"
        assert exposed["config_dict"]["current_value"] == {"key": "value"}

    def test_multiple_agents_independent_exposed_params(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test that different agent instances have independent exposed params."""
        agent1 = TestAgentWithExposedParams()
        agent2 = TestAgentWithExposedParams()

        # Modify agent1
        agent1.max_iterations = 15

        # Verify agent2 is unaffected
        assert agent1.max_iterations == 15
        assert agent2.max_iterations == 5

        exposed1 = agent1.get_exposed_params(include_values=True)
        exposed2 = agent2.get_exposed_params(include_values=True)

        assert exposed1["max_iterations"]["current_value"] == 15
        assert exposed2["max_iterations"]["current_value"] == 5
