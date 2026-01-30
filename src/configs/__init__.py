"""Configuration loader for multi-model versioning."""
import importlib


def load_config(version: str):
    """
    Load configuration for a specific model version.

    Args:
        version: Model version string ('v1', 'v2', 'v3', 'v4')

    Returns:
        Config module with version-specific settings
    """
    configs = {
        'v1': 'configs.v1_config',
        'v2': 'configs.v2_config',
        'v4': 'configs.v4_config',
    }

    if version not in configs:
        raise ValueError(f"Unknown version: {version}. Available: {list(configs.keys())}")

    return importlib.import_module(configs[version])


def get_available_versions():
    """Return list of available model versions."""
    return ['v1', 'v2', 'v4']
