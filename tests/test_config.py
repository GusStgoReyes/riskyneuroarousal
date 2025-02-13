import pytest
from riskyneuroarousal.config import Config

# FILE: src/riskyneuroarousal/test_config.py


def test_config_instantiation():
    try:
        config = Config(file_path='tests/test_config.json')
        assert True
    except Exception as e:
        pytest.fail(f"Instantiation failed: {e}")

def test_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        Config(file_path='non_existent_config.json')

def test_config_with_additional_kwargs():
    config = Config(file_path='tests/test_config.json', additional_param='value')
    assert hasattr(config, 'additional_param')
    assert config.additional_param == 'value'