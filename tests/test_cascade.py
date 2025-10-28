import pytest

from src.models.cascade import cascade_coverage


def test_cascade_coverage():
    coverage = cascade_coverage([True, False, True, True])
    assert pytest.approx(coverage) == 0.75
