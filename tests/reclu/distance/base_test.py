import pandas as pd
import pytest

from reidfo.reclu.distance.base import BaseDistance
from reidfo.reclu.distance.jaccard import JaccardDistance


def test_base_distance_rejects_non_datetime_index():
    labels = {"a": pd.Series([0, 1, 0], index=["x", "y", "z"])}
    with pytest.raises(ValueError):
        JaccardDistance(labels)


def test_base_distance_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseDistance({})
