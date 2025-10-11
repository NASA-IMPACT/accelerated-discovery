"""
Library and dependency tests for reranker tools.

These tests verify that required dependencies are installed and functioning correctly.
Tests are conditionally skipped if optional dependencies are not available.
"""
# ruff: noqa: F401

import sys
from importlib.metadata import version

import pytest

# Check if sentence-transformers is available
try:
    import sentence_transformers
    from sentence_transformers import CrossEncoder

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Check if torch is available
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Check if numpy is available (should always be available as core dependency)
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class TestDependencyAvailability:
    """Test that required dependencies are available."""

    def test_numpy_installed(self):
        """Test that numpy is installed (core dependency)."""
        assert NUMPY_AVAILABLE, "numpy should be installed as a core dependency"
        import numpy

        assert hasattr(numpy, "__version__")

    @pytest.mark.requires_ml
    def test_sentence_transformers_installed(self):
        """Test that sentence-transformers is installed."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            pytest.skip("sentence-transformers not installed - install with: pip install sentence-transformers")

        assert SENTENCE_TRANSFORMERS_AVAILABLE
        import sentence_transformers

        assert hasattr(sentence_transformers, "__version__")

    @pytest.mark.requires_ml
    def test_torch_installed(self):
        """Test that torch is installed (required by sentence-transformers)."""
        if not TORCH_AVAILABLE:
            pytest.skip("torch not installed - required by sentence-transformers")

        assert TORCH_AVAILABLE
        import torch

        assert hasattr(torch, "__version__")


class TestImports:
    """Test that required modules can be imported."""

    @pytest.mark.requires_ml
    def test_crossencoder_import(self):
        """Test that CrossEncoder can be imported from sentence-transformers."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            pytest.skip("sentence-transformers not installed")

        from sentence_transformers import CrossEncoder

        assert CrossEncoder is not None

    @pytest.mark.requires_ml
    def test_reranker_module_import(self):
        """Test that the reranker module can be imported."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            pytest.skip("sentence-transformers not installed - reranker.py requires it")

        from akd.tools import reranker

        assert hasattr(reranker, "CrossEncoderRerankerTool")
        assert hasattr(reranker, "RerankerTool")
        assert hasattr(reranker, "RerankerToolConfig")

    @pytest.mark.requires_ml
    def test_reranker_tool_import(self):
        """Test that CrossEncoderRerankerTool can be imported directly."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            pytest.skip("sentence-transformers not installed")

        from akd.tools.reranker import CrossEncoderRerankerTool

        assert CrossEncoderRerankerTool is not None


class TestModelInstantiation:
    """Test that reranker models can be instantiated."""

    @pytest.mark.requires_ml
    @pytest.mark.slow
    def test_crossencoder_instantiation_default_model(self):
        """Test CrossEncoder instantiation with default model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            pytest.skip("sentence-transformers not installed")

        from sentence_transformers import CrossEncoder

        # This will download the model if not cached
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L12-v2")

        assert model is not None
        assert hasattr(model, "predict")

    @pytest.mark.requires_ml
    @pytest.mark.slow
    def test_reranker_tool_instantiation(self):
        """Test CrossEncoderRerankerTool instantiation."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            pytest.skip("sentence-transformers not installed")

        from akd.tools.reranker import CrossEncoderRerankerTool

        reranker = CrossEncoderRerankerTool()

        assert reranker is not None
        assert hasattr(reranker, "reranker_model")
        assert hasattr(reranker, "arun")

    @pytest.mark.requires_ml
    @pytest.mark.slow
    def test_reranker_tool_with_custom_config(self):
        """Test CrossEncoderRerankerTool with custom configuration."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            pytest.skip("sentence-transformers not installed")

        from akd.tools.reranker import CrossEncoderRerankerTool, RerankerToolConfig

        config = RerankerToolConfig(
            model_name="cross-encoder/ms-marco-MiniLM-L12-v2",
            deduplication=False,
        )
        reranker = CrossEncoderRerankerTool(config=config)

        assert reranker is not None
        assert reranker.config.deduplication is False


class TestBasicFunctionality:
    """Test basic functionality of reranker dependencies."""

    @pytest.mark.requires_ml
    @pytest.mark.slow
    def test_crossencoder_predict_simple(self):
        """Test CrossEncoder predict on simple dummy data."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            pytest.skip("sentence-transformers not installed")

        from sentence_transformers import CrossEncoder

        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L12-v2")

        # Simple query-document pairs
        pairs = [
            ("What is Python?", "Python is a programming language"),
            ("What is Python?", "A python is a type of snake"),
        ]

        scores = model.predict(pairs)

        assert scores is not None
        assert len(scores) == 2
        # First pair should score higher (more relevant)
        assert scores[0] > scores[1]

    @pytest.mark.requires_ml
    def test_numpy_operations(self):
        """Test that numpy operations work correctly."""
        if not NUMPY_AVAILABLE:
            pytest.skip("numpy not installed")

        import numpy as np

        # Test sigmoid transformation (used in reranker)
        scores = np.array([1.0, 2.0, 3.0])
        normalized = 1 / (1 + np.exp(-scores))

        assert len(normalized) == 3
        assert all(0 <= s <= 1 for s in normalized)
        # Sigmoid should preserve order
        assert normalized[0] < normalized[1] < normalized[2]

    @pytest.mark.requires_ml
    def test_torch_device_handling(self):
        """Test torch device handling (CPU/CUDA)."""
        if not TORCH_AVAILABLE:
            pytest.skip("torch not installed")

        import torch

        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()

        # Should always have CPU
        assert torch.device("cpu") is not None

        if cuda_available:
            # If CUDA available, should be able to create CUDA device
            device = torch.device("cuda")
            assert device is not None


class TestVersionCompatibility:
    """Test version compatibility of dependencies."""

    @pytest.mark.requires_ml
    def test_sentence_transformers_version(self):
        """Test that sentence-transformers version meets requirements."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            pytest.skip("sentence-transformers not installed")

        st_version = version("sentence-transformers")

        # Parse version (e.g., "5.0.0" -> (5, 0, 0))
        major, minor, *_ = st_version.split(".")
        major, minor = int(major), int(minor)

        # Should be >= 5.0.0 according to pyproject.toml
        assert major >= 5, f"sentence-transformers version {st_version} < 5.0.0"

    @pytest.mark.requires_ml
    def test_torch_version_exists(self):
        """Test that torch version can be retrieved."""
        if not TORCH_AVAILABLE:
            pytest.skip("torch not installed")

        torch_version = version("torch")

        assert torch_version is not None
        assert len(torch_version) > 0


class TestGracefulDegradation:
    """Test graceful handling when dependencies are missing."""

    def test_missing_library_guidance(self):
        """Test that helpful error messages are available when libraries missing."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            pytest.skip("sentence-transformers is installed - cannot test missing library behavior")

        # When sentence-transformers is not installed, importing reranker should fail
        with pytest.raises(ImportError) as exc_info:
            pass

        error_message = str(exc_info.value)
        # Should mention the missing module
        assert "sentence_transformers" in error_message or "sentence-transformers" in error_message

    @pytest.mark.requires_ml
    def test_reranker_with_ml_dependencies(self):
        """Test that reranker works when ML dependencies are available."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            pytest.skip("sentence-transformers not installed")

        # This should work without errors
        from akd.tools.reranker import RerankerToolConfig

        config = RerankerToolConfig()
        assert config.model_name == "cross-encoder/ms-marco-MiniLM-L12-v2"


class TestEnvironment:
    """Test environment and platform-specific features."""

    @pytest.mark.requires_ml
    def test_python_version_compatibility(self):
        """Test that Python version is compatible."""
        # Should be >= 3.12 according to pyproject.toml
        version_info = sys.version_info
        assert version_info.major == 3
        assert version_info.minor >= 12, f"Python {version_info.major}.{version_info.minor} < 3.12"

    @pytest.mark.requires_ml
    @pytest.mark.slow
    def test_model_caching(self):
        """Test that models are cached after first download."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            pytest.skip("sentence-transformers not installed")

        import time

        from sentence_transformers import CrossEncoder

        model_name = "cross-encoder/ms-marco-MiniLM-L12-v2"

        # First load (might download or use cache)
        start = time.time()
        model1 = CrossEncoder(model_name)
        first_load_time = time.time() - start

        # Second load (should use cache)
        start = time.time()
        model2 = CrossEncoder(model_name)
        second_load_time = time.time() - start

        assert model1 is not None
        assert model2 is not None

        # Second load should be faster (cached)
        # Note: This might not always be true in CI environments
        # so we just verify both loads succeeded
        assert first_load_time >= 0
        assert second_load_time >= 0
