import pytest
from unittest.mock import patch
from src.model_utils import load_model

def test_TC01_model_loads():
    model = load_model()
    assert model is not None

def test_TC02_model_has_predict():
    model = load_model()
    assert hasattr(model, "predict")

def test_TC03_model_accepts_input():
    model = load_model()
    sample_input = [[90, 42, 43, 21, 82, 6.5, 203]]
    output = model.predict(sample_input)
    assert output is not None

def test_TC04_model_output_type():
    model = load_model()
    sample_input = [[90, 42, 43, 21, 82, 6.5, 203]]
    output = model.predict(sample_input)
    assert isinstance(output[0], str)

def test_TC05_model_load_failure():
    with patch("src.model_utils.os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            load_model()
