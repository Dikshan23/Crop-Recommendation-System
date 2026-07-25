import pytest
from src.predict import predict_crop


def test_TC01_valid_crop_prediction():
    crop, confidence, probabilities = predict_crop(
        85, 50, 40, 22, 82, 7.0, 230
    )

    assert crop is not None
    assert isinstance(crop, str)
    assert isinstance(confidence, float)
    assert isinstance(probabilities, dict)


def test_TC02_nitrogen_out_of_range():
    with pytest.raises(ValueError):
        predict_crop(
            200, 50, 40, 22, 82, 7.0, 230
        )


def test_TC03_ph_out_of_range():
    with pytest.raises(ValueError):
        predict_crop(
            85, 50, 40, 22, 82, 12.5, 230
        )