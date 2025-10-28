import pytest

from src.models.model_zoo import ModelZoo


def test_unknown_model_raises():
    zoo = ModelZoo()
    with pytest.raises(KeyError):
        zoo.load_text_model("unknown", "fp32", "cpu")
