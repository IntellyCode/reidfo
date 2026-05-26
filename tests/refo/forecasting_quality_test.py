import pandas as pd
import pytest

from reidfo.refo.forecasting_quality import ForecastingQuality


def _series_inputs():
    train = pd.Series([0, 0, 0, 1, 1])
    expected = pd.Series([0, 1, 0, 1, 0])
    forecasted = pd.Series([0, 1, 0, 0, 0])
    return train, expected, forecasted


def test_series_input_returns_three_metrics():
    train, expected, forecasted = _series_inputs()
    fq = ForecastingQuality(train, expected, forecasted)
    result = fq.get_forecasting_stats()
    assert list(result.columns) == ["Model Accuracy", "MCR Accuracy", "Random Accuracy"]
    assert len(result) == 1


def test_model_accuracy_is_correct():
    train, expected, forecasted = _series_inputs()
    fq = ForecastingQuality(train, expected, forecasted)
    result = fq.get_forecasting_stats()
    # 4 out of 5 match
    assert result.loc[0, "Model Accuracy"] == pytest.approx(0.8)


def test_mixed_types_raise():
    train = pd.Series([0, 1, 0])
    expected = pd.DataFrame({"a": [0, 1, 0]})
    forecasted = pd.Series([0, 0, 0])
    with pytest.raises(TypeError):
        ForecastingQuality(train, expected, forecasted)


def test_series_index_mismatch_raises():
    train = pd.Series([0, 1, 0])
    expected = pd.Series([0, 1, 0])
    forecasted = pd.Series([0, 0, 0], index=[10, 11, 12])
    with pytest.raises(ValueError):
        ForecastingQuality(train, expected, forecasted)


def test_dataframe_input_returns_per_row_metrics():
    idx = ["A", "B"]
    train = pd.DataFrame([[0, 0, 1, 1], [1, 1, 0, 0]], index=idx)
    expected = pd.DataFrame([[0, 1, 0, 1], [1, 0, 1, 0]], index=idx)
    forecasted = pd.DataFrame([[0, 1, 0, 0], [1, 0, 0, 0]], index=idx)
    fq = ForecastingQuality(train, expected, forecasted)
    result = fq.get_forecasting_stats()
    assert list(result.index) == idx
    assert list(result.columns) == ["Model Accuracy", "MCR Accuracy", "Random Accuracy"]
