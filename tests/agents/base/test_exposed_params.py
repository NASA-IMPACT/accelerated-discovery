"""Test cases for exposed parameter functionality."""

from akd._base import exposed_param

from .conftest import DummyAgentWithExposedConfig, TestInstructorBaseAgent


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

        exposed_list = agent.get_exposed_params(include_values=False)
        exposed = {p.name: p for p in exposed_list}  # Convert to dict for easier testing

        # Verify all exposed params are present
        assert "clarification_prompt" in exposed
        assert "max_iterations" in exposed
        assert "temperature_override" in exposed
        assert "computed_value" in exposed

        # Verify metadata structure
        assert exposed["clarification_prompt"].description == "System prompt for query clarification"
        assert exposed["clarification_prompt"].type_ == "str"
        assert exposed["clarification_prompt"].editable is True

        # Verify extra metadata
        assert exposed["max_iterations"].extra is not None
        assert exposed["max_iterations"].extra["min"] == 1
        assert exposed["max_iterations"].extra["max"] == 20

        # Verify no values included
        assert exposed["clarification_prompt"].current_value is None

    def test_get_exposed_params_with_values(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test get_exposed_params returns metadata with current values."""
        agent = TestAgentWithExposedParams()

        exposed_list = agent.get_exposed_params(include_values=True)
        exposed = {p.name: p for p in exposed_list}

        # Verify values are included
        assert exposed["clarification_prompt"].current_value == "Default clarification prompt"
        assert exposed["max_iterations"].current_value == 5
        assert exposed["temperature_override"].current_value == 0.7

    def test_exposed_param_auto_description_from_docstring(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test that exposed_param uses docstring when no description provided."""
        agent = TestAgentWithExposedParams()

        exposed_list = agent.get_exposed_params()
        exposed = {p.name: p for p in exposed_list}

        # temperature_override has no explicit description, should use docstring
        assert exposed["temperature_override"].description == "Temperature override for this agent"

    def test_exposed_param_editable_detection(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test that editable flag is auto-detected from setter presence."""
        agent = TestAgentWithExposedParams()

        exposed_list = agent.get_exposed_params()
        exposed = {p.name: p for p in exposed_list}

        # Parameters with setters should be editable
        assert exposed["clarification_prompt"].editable is True
        assert exposed["max_iterations"].editable is True
        assert exposed["temperature_override"].editable is True

        # Parameter without setter should be read-only
        assert exposed["computed_value"].editable is False

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
        exposed_list = agent.get_exposed_params(include_values=True)
        exposed = {p.name: p for p in exposed_list}
        assert exposed["clarification_prompt"].current_value == "New prompt"
        assert exposed["max_iterations"].current_value == 10

    def test_exposed_param_extra_metadata(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test that extra metadata is properly stored and retrieved."""
        agent = TestAgentWithExposedParams()

        exposed_list = agent.get_exposed_params()
        exposed = {p.name: p for p in exposed_list}

        # max_iterations has extra metadata (min, max)
        assert exposed["max_iterations"].extra is not None
        assert exposed["max_iterations"].extra["min"] == 1
        assert exposed["max_iterations"].extra["max"] == 20

        # clarification_prompt has no extra metadata (empty dict by default)
        assert not exposed["clarification_prompt"].extra

    def test_exposed_param_type_inference(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test that types are correctly inferred from current values."""
        agent = TestAgentWithExposedParams()

        exposed_list = agent.get_exposed_params()
        exposed = {p.name: p for p in exposed_list}

        assert exposed["clarification_prompt"].type_ == "str"
        assert exposed["max_iterations"].type_ == "int"
        assert exposed["temperature_override"].type_ == "float"
        assert exposed["computed_value"].type_ == "str"

    def test_regular_properties_not_exposed(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test that regular @property decorated attributes are not exposed."""
        agent = TestAgentWithExposedParams()

        exposed_list = agent.get_exposed_params()
        exposed = {p.name: p for p in exposed_list}

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
        exposed_list = agent.get_exposed_params(include_values=True)
        exposed = {p.name: p for p in exposed_list}

        assert exposed["config_dict"].type_ == "dict"
        assert exposed["config_dict"].current_value == {"key": "value"}

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

        exposed1_list = agent1.get_exposed_params(include_values=True)
        exposed1 = {p.name: p for p in exposed1_list}
        exposed2_list = agent2.get_exposed_params(include_values=True)
        exposed2 = {p.name: p for p in exposed2_list}

        assert exposed1["max_iterations"].current_value == 15
        assert exposed2["max_iterations"].current_value == 5

    def test_exposed_param_type_validation(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test that setters automatically validate types."""
        agent = TestAgentWithExposedParams()

        # Valid assignments should work
        agent.max_iterations = 10
        assert agent.max_iterations == 10

        # Invalid type should raise TypeError
        import pytest

        with pytest.raises(TypeError, match="expects int, got str"):
            agent.max_iterations = "invalid"

        with pytest.raises(TypeError, match="expects float, got str"):
            agent.temperature_override = "invalid"

        with pytest.raises(TypeError, match="expects str, got int"):
            agent.clarification_prompt = 123

    def test_config_schema_property_access_patterns(
        self,
        mock_openai_client,
        mock_instructor_client,
    ):
        """Test that config schema fields support all 3 access patterns."""
        agent = DummyAgentWithExposedConfig()

        # Verify exposed params include config_ prefixed fields
        exposed_list = agent.get_exposed_params()
        exposed = {p.name: p for p in exposed_list}
        assert "config_topic_steer" in exposed, "config_topic_steer should be exposed"
        assert "config_temperature" in exposed, "config_temperature should be exposed"

        # Test 1: Set via setattr with config_ prefixed property name
        setattr(agent, "config_topic_steer", "new_value")

        # Verify access pattern 1: obj.config.topic_steer (direct config access)
        assert agent.config.topic_steer == "new_value", "obj.config.topic_steer should work"

        # Verify access pattern 2: obj.config_topic_steer (prefixed property)
        assert agent.config_topic_steer == "new_value", "obj.config_topic_steer should work"

        # Verify access pattern 3: Check if obj.topic_steer exists (unprefixed)
        # This may or may not exist depending on implementation
        try:
            value = getattr(agent, "topic_steer", None)
            if value is not None:
                assert value == "new_value", "obj.topic_steer should match if it exists"
        except AttributeError:
            pass  # It's OK if unprefixed version doesn't exist

        # Test 2: Set via direct config access and verify property updates
        agent.config.topic_steer = "direct_value"
        assert agent.config_topic_steer == "direct_value", "Property should reflect config changes"

        # Test 3: Set via property assignment and verify config updates
        agent.config_topic_steer = "property_value"
        assert agent.config.topic_steer == "property_value", "Config should reflect property changes"

        # Test 4: Verify temperature field with setattr
        setattr(agent, "config_temperature", 0.9)
        assert agent.config.temperature == 0.9, "setattr should work for config_temperature"
        assert agent.config_temperature == 0.9, "config_temperature property should work"

        # Test 5: Verify all 3 patterns are consistent
        test_value = "consistency_test"
        agent.config_topic_steer = test_value
        assert agent.config.topic_steer == test_value, "All access patterns should be consistent"
        assert agent.config_topic_steer == test_value, "All access patterns should be consistent"


class TestParameterDiscovery:
    """Tests for parameter discovery with nested components via scan_runtime."""

    def test_scan_runtime_disabled(self, exposed_simple_agent):
        """Test that scan_runtime=False doesn't discover runtime components."""
        params = exposed_simple_agent.get_exposed_params(scan_runtime=False)
        param_names = {p.name for p in params}

        # Should NOT include runtime-discovered component fields
        assert "component.value" not in param_names

    def test_scan_runtime_enabled(self, exposed_simple_agent):
        """Test that scan_runtime=True discovers nested component fields."""
        params = exposed_simple_agent.get_exposed_params(scan_runtime=True)
        param_names = {p.name for p in params}

        # Should include runtime-discovered component fields
        assert "component.value" in param_names

    def test_nested_config_discovery(self, exposed_complex_agent):
        """Test runtime scanning discovers deeply nested config parameters."""
        params = exposed_complex_agent.get_exposed_params(scan_runtime=True)
        param_names = {p.name for p in params}

        assert "component.name" in param_names
        assert "component.config.temperature" in param_names
        assert "component.config.max_tokens" in param_names

    def test_multiple_components_discovery(self, exposed_multi_agent):
        """Test runtime scanning with multiple nested components."""
        params = exposed_multi_agent.get_exposed_params(scan_runtime=True)
        param_names = {p.name for p in params}

        assert "simple.value" in param_names
        assert "complex.name" in param_names
        assert "complex.config.temperature" in param_names

    def test_metadata_extraction(self, exposed_complex_agent):
        """Test that metadata is correctly extracted from Exposed annotations."""
        params = exposed_complex_agent.get_exposed_params(scan_runtime=True)
        param_dict = {p.name: p for p in params}

        # Check descriptions
        assert param_dict["component.name"].description == "Component name"
        assert param_dict["component.config.temperature"].description == "Temperature parameter"

        # Check types
        assert param_dict["component.name"].type_ == "str"
        assert param_dict["component.config.temperature"].type_ == "float"
        assert param_dict["component.config.max_tokens"].type_ == "int"

    def test_include_values_false(self, exposed_simple_agent):
        """Test that include_values=False doesn't populate current_value."""
        params = exposed_simple_agent.get_exposed_params(
            scan_runtime=True,
            include_values=False,
        )

        for p in params:
            assert p.current_value is None

    def test_include_values_true(self, exposed_complex_agent):
        """Test that include_values=True populates current_value."""
        params = exposed_complex_agent.get_exposed_params(
            scan_runtime=True,
            include_values=True,
        )
        param_dict = {p.name: p for p in params}

        assert param_dict["component.name"].current_value == "component"
        assert param_dict["component.config.temperature"].current_value == 0.7
        assert param_dict["component.config.max_tokens"].current_value == 100


class TestAttributeAccess:
    """Tests for dict-like attribute access ([], get()) with nested components."""

    # __getitem__ tests
    def test_getitem_simple_path(self, exposed_simple_agent):
        """Test __getitem__ with simple path."""
        value = exposed_simple_agent["component.value"]
        assert value == "default"

    def test_getitem_nested_path(self, exposed_complex_agent):
        """Test __getitem__ with nested path."""
        value = exposed_complex_agent["component.config.temperature"]
        assert value == 0.7

    def test_getitem_intermediate_object(self, exposed_complex_agent):
        """Test __getitem__ can access intermediate objects."""
        from .conftest import ExposedNestedConfig

        config = exposed_complex_agent["component.config"]
        assert isinstance(config, ExposedNestedConfig)
        assert config.temperature == 0.7

    def test_getitem_multiple_paths(self, exposed_multi_agent):
        """Test __getitem__ with multiple component paths."""
        assert exposed_multi_agent["simple.value"] == "default"
        assert exposed_multi_agent["complex.name"] == "component"
        assert exposed_multi_agent["complex.config.temperature"] == 0.7

    def test_getitem_missing_raises(self, exposed_simple_agent):
        """Test that __getitem__ raises AttributeError for missing path."""
        import pytest

        with pytest.raises(AttributeError):
            _ = exposed_simple_agent["nonexistent.path"]

    def test_getitem_partial_missing_raises(self, exposed_complex_agent):
        """Test that __getitem__ raises AttributeError for partial missing path."""
        import pytest

        with pytest.raises(AttributeError):
            _ = exposed_complex_agent["component.missing.field"]

    # __setitem__ tests
    def test_setitem_simple_path(self, exposed_simple_agent):
        """Test __setitem__ with simple path."""
        exposed_simple_agent["component.value"] = "new_value"
        assert exposed_simple_agent.component.value == "new_value"

    def test_setitem_nested_path(self, exposed_complex_agent):
        """Test __setitem__ with nested path."""
        exposed_complex_agent["component.config.temperature"] = 0.9
        assert exposed_complex_agent.component.config.temperature == 0.9

    def test_setitem_multiple_updates(self, exposed_complex_agent):
        """Test multiple __setitem__ operations."""
        exposed_complex_agent["component.name"] = "updated"
        exposed_complex_agent["component.config.temperature"] = 0.5
        exposed_complex_agent["component.config.max_tokens"] = 200

        assert exposed_complex_agent.component.name == "updated"
        assert exposed_complex_agent.component.config.temperature == 0.5
        assert exposed_complex_agent.component.config.max_tokens == 200

    def test_setitem_missing_raises(self, exposed_simple_agent):
        """Test that __setitem__ raises AttributeError for missing path."""
        import pytest

        with pytest.raises(AttributeError):
            exposed_simple_agent["nonexistent.path"] = "value"

    # get() method tests
    def test_get_existing_path(self, exposed_simple_agent):
        """Test get() method returns value for existing path."""
        value = exposed_simple_agent.get("component.value", "fallback")
        assert value == "default"  # The actual value, not the fallback

    def test_get_nested_path(self, exposed_complex_agent):
        """Test get() method with nested path."""
        value = exposed_complex_agent.get("component.config.temperature", 0.0)
        assert value == 0.7

    def test_get_missing_with_default(self, exposed_simple_agent):
        """Test get() method returns default for missing path."""
        value = exposed_simple_agent.get("nonexistent.path", "fallback")
        assert value == "fallback"

    def test_get_missing_none_default(self, exposed_simple_agent):
        """Test get() method with None as default."""
        value = exposed_simple_agent.get("nonexistent.path")
        assert value is None

    def test_get_vs_getitem_missing_key(self, exposed_complex_agent):
        """Test that get() doesn't raise but __getitem__ does for missing keys."""
        import pytest

        # get() returns default
        value = exposed_complex_agent.get("missing.key", "default")
        assert value == "default"

        # __getitem__ raises
        with pytest.raises(AttributeError):
            _ = exposed_complex_agent["missing.key"]


class TestIntegration:
    """Integration tests for mixed operations and round-trip scenarios."""

    def test_round_trip_simple(self, exposed_simple_agent):
        """Test setting and getting via dict interface."""
        exposed_simple_agent["component.value"] = "updated"

        # Via dict interface
        assert exposed_simple_agent["component.value"] == "updated"
        # Via direct access
        assert exposed_simple_agent.component.value == "updated"

    def test_round_trip_nested(self, exposed_complex_agent):
        """Test round-trip with nested paths."""
        exposed_complex_agent["component.name"] = "new_name"
        exposed_complex_agent["component.config.temperature"] = 0.85

        # Via dict interface
        assert exposed_complex_agent["component.name"] == "new_name"
        assert exposed_complex_agent["component.config.temperature"] == 0.85

        # Via direct access
        assert exposed_complex_agent.component.name == "new_name"
        assert exposed_complex_agent.component.config.temperature == 0.85

    def test_round_trip_multiple_components(self, exposed_multi_agent):
        """Test round-trip with multiple components."""
        exposed_multi_agent["simple.value"] = "simple_updated"
        exposed_multi_agent["complex.name"] = "complex_updated"
        exposed_multi_agent["complex.config.temperature"] = 0.95

        assert exposed_multi_agent["simple.value"] == "simple_updated"
        assert exposed_multi_agent["complex.name"] == "complex_updated"
        assert exposed_multi_agent["complex.config.temperature"] == 0.95

    def test_modify_via_dict_check_via_params(self, exposed_complex_agent):
        """Test modifying via dict interface and checking via get_exposed_params."""
        # Modify via dict interface
        exposed_complex_agent["component.config.temperature"] = 0.99

        # Check via get_exposed_params with values
        params = exposed_complex_agent.get_exposed_params(
            scan_runtime=True,
            include_values=True,
        )
        temp_param = next(p for p in params if p.name == "component.config.temperature")

        assert temp_param.current_value == 0.99

    def test_get_for_safe_access(self, exposed_multi_agent):
        """Test using get() for safe access to potentially missing paths."""
        # Existing paths
        assert exposed_multi_agent.get("simple.value") == "default"
        assert exposed_multi_agent.get("complex.name") == "component"

        # Non-existing paths don't raise
        assert exposed_multi_agent.get("nonexistent") is None
        assert exposed_multi_agent.get("also.missing", "fallback") == "fallback"

    def test_mixed_access_methods(self, exposed_complex_agent):
        """Test mixing direct access, dict access, and get() method."""
        # Set via dict interface
        exposed_complex_agent["component.name"] = "dict_set"

        # Get via direct access
        assert exposed_complex_agent.component.name == "dict_set"

        # Get via get() method
        assert exposed_complex_agent.get("component.name") == "dict_set"

        # Set via direct access
        exposed_complex_agent.component.config.temperature = 0.6

        # Get via dict interface
        assert exposed_complex_agent["component.config.temperature"] == 0.6
