"""
Unit tests for FloodSense AI Model Inference & Geospatial Metadata Preservation
"""

from pathlib import Path
import numpy as np
import rasterio
import pytest

from models.predict import FloodPredictor


@pytest.fixture(scope="module")
def sample_s1_path():
    base_dir = Path(__file__).resolve().parent.parent.parent
    sample = base_dir / "data" / "raw" / "S1Hand" / "USA_1010394_S1Hand.tif"
    if not sample.exists():
        # Fallback to any present S1Hand file
        files = list((base_dir / "data" / "raw" / "S1Hand").glob("*_S1Hand.tif"))
        assert len(files) > 0, "No S1Hand GeoTIFF files found in data/raw/S1Hand"
        return files[0]
    return sample


@pytest.fixture(scope="module")
def checkpoint_path():
    base_dir = Path(__file__).resolve().parent.parent.parent
    ckpt = base_dir / "models" / "checkpoints" / "unet_baseline_best.pt"
    assert ckpt.exists(), f"Model checkpoint not found: {ckpt}"
    return ckpt


def test_predictor_loads_checkpoint_and_frozen_stats(checkpoint_path):
    """Verifies that the predictor loads without recomputing normalization statistics."""
    predictor = FloodPredictor(checkpoint_path=checkpoint_path)
    assert predictor.model is not None
    assert predictor.threshold == 0.5
    # Check that frozen training statistics are present and non-zero
    assert predictor.vv_mean < 0.0  # Typically around -10.45 dB
    assert predictor.vv_std > 0.0   # Typically around 4.06 dB
    assert predictor.vh_mean < 0.0  # Typically around -17.32 dB
    assert predictor.vh_std > 0.0   # Typically around 4.79 dB


def test_inference_output_properties(checkpoint_path, sample_s1_path):
    """Verifies output dimensions, dtypes, and finite probability ranges."""
    predictor = FloodPredictor(checkpoint_path=checkpoint_path)

    with rasterio.open(sample_s1_path) as src:
        s1_data = src.read()

    res = predictor.predict_array(s1_data)

    probs = res["probabilities"]
    mask = res["binary_mask"]
    valid = res["valid_mask"]

    # Dimensions
    assert probs.shape == (512, 512)
    assert mask.shape == (512, 512)
    assert valid.shape == (512, 512)

    # Probability bounds for valid pixels
    valid_probs = probs[valid]
    assert len(valid_probs) > 0
    assert np.all(valid_probs >= 0.0)
    assert np.all(valid_probs <= 1.0)
    assert np.all(np.isfinite(valid_probs))

    # Mask categories
    unique_mask_vals = set(np.unique(mask))
    assert unique_mask_vals.issubset({0, 1, 255})


def test_geospatial_metadata_preservation(checkpoint_path, sample_s1_path, tmp_path):
    """Verifies that CRS, affine transform, and bounding box are preserved in output GeoTIFFs."""
    predictor = FloodPredictor(checkpoint_path=checkpoint_path)

    res = predictor.predict_geotiff(
        input_path=sample_s1_path,
        output_dir=tmp_path,
        save_rasters=True,
    )

    prob_file = Path(res["prob_geotiff"])
    mask_file = Path(res["mask_geotiff"])

    assert prob_file.exists()
    assert mask_file.exists()

    with rasterio.open(sample_s1_path) as src_in, rasterio.open(prob_file) as src_prob, rasterio.open(mask_file) as src_mask:
        # Check CRS
        assert src_prob.crs == src_in.crs
        assert src_mask.crs == src_in.crs

        # Check Dimensions
        assert src_prob.width == src_in.width
        assert src_prob.height == src_in.height
        assert src_mask.width == src_in.width
        assert src_mask.height == src_in.height

        # Check Affine Transform
        assert np.isclose(src_prob.transform.a, src_in.transform.a, atol=1e-6)
        assert np.isclose(src_prob.transform.c, src_in.transform.c, atol=1e-6)
        assert np.isclose(src_prob.transform.e, src_in.transform.e, atol=1e-6)
        assert np.isclose(src_prob.transform.f, src_in.transform.f, atol=1e-6)


def test_inference_reproducibility(checkpoint_path, sample_s1_path):
    """Verifies deterministic outputs across multiple forward passes on the same input."""
    predictor = FloodPredictor(checkpoint_path=checkpoint_path)

    with rasterio.open(sample_s1_path) as src:
        s1_data = src.read()

    res1 = predictor.predict_array(s1_data)
    res2 = predictor.predict_array(s1_data)

    p1 = res1["probabilities"]
    p2 = res2["probabilities"]

    # Valid mask should match exactly
    assert np.array_equal(res1["valid_mask"], res2["valid_mask"])

    # Probabilities on valid pixels should be strictly identical
    val = res1["valid_mask"]
    assert np.allclose(p1[val], p2[val], atol=1e-6)
