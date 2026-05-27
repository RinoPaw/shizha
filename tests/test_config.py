"""Smoke tests for configuration loading."""

from anti_fraud_explorer.config import settings


def test_settings_loads():
    """Settings object can be instantiated."""
    assert settings is not None


def test_dataset_path_exists():
    """Dataset path points to a real location."""
    path = settings.dataset_path
    assert path is not None
    assert path.parent.exists()


def test_port_is_integer():
    """Port is a positive integer."""
    assert isinstance(settings.port, int)
    assert settings.port > 0


def test_host_is_non_empty():
    """Host is a non-empty string."""
    assert isinstance(settings.host, str)
    assert len(settings.host) > 0


def test_ai_model_has_default():
    """AI model has a default value."""
    assert settings.ai_model, "AI_MODEL should have a default"


def test_debug_defaults_false():
    """Debug is off by default."""
    assert settings.debug is False
