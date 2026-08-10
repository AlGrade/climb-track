import numpy as np

from climbtrack.backends.sapiens2 import PosePrediction, _average_scales, _scaled_dimension


def test_scaled_dimensions_stay_patch_aligned() -> None:
    assert _scaled_dimension(1024, 1.125) == 1152
    assert _scaled_dimension(768, 1.125) == 864
    assert _scaled_dimension(100, 1.01) % 16 == 0


def test_multiscale_predictions_are_arithmetic_means() -> None:
    first = PosePrediction(
        coordinates=np.zeros((308, 2), dtype=np.float32),
        confidence=np.full(308, 0.2, dtype=np.float32),
    )
    second = PosePrediction(
        coordinates=np.full((308, 2), 2.0, dtype=np.float32),
        confidence=np.full(308, 0.8, dtype=np.float32),
    )

    result = _average_scales([[first], [second]])[0]

    np.testing.assert_allclose(result.coordinates, 1.0)
    np.testing.assert_allclose(result.confidence, 0.5)
