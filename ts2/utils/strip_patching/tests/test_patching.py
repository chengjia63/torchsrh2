import builtins
from types import SimpleNamespace

import numpy as np

from ts2.utils.strip_patching import patching as sp


def _fake_dcmread_factory(pixel_arrays):
    def _fake_dcmread(path):
        return SimpleNamespace(pixel_array=pixel_arrays[str(path)])

    return _fake_dcmread


def _patch_coords(patch_dict):
    coords = set()
    for name in patch_dict:
        coord_str = name.rsplit("-", 1)[1]
        y_str, x_str = coord_str.split("_", 1)
        coords.add((int(y_str), int(x_str)))
    return coords


def test_derived_substrip_layout_respects_arbitrary_patch_origin():
    substrip_start, substrip_stride = sp._derived_substrip_layout(
        patch_size=300,
        patch_stride=300,
        patch_start=900,
        substrip_size=1000,
    )

    assert substrip_start == 300
    assert substrip_stride == 900


def test_generate_paired_strip_patches_covers_arbitrary_patch_start(monkeypatch):
    pixel_arrays = {
        "ch2.dcm": np.zeros((2200, 1000), dtype=np.uint16),
        "ch3.dcm": np.ones((2200, 1000), dtype=np.uint16),
    }
    monkeypatch.setattr(sp.pyd, "dcmread", _fake_dcmread_factory(pixel_arrays))

    patches = sp.generate_paired_strip_patches(
        "ch2.dcm",
        "ch3.dcm",
        patch_size=300,
        patch_stride=300,
        patch_start=(900, 0),
        substrip_size=1000,
        register=False,
        patch_processor=None,
    )

    expected_coords = {(y, x) for y in (900, 1200, 1500, 1800) for x in (0, 300, 600)}
    assert _patch_coords(patches) == expected_coords


def test_generate_paired_strip_patches_covers_tail_of_long_strip(monkeypatch):
    pixel_arrays = {
        "ch2.dcm": np.zeros((2000, 1000), dtype=np.uint16),
        "ch3.dcm": np.ones((2000, 1000), dtype=np.uint16),
    }
    monkeypatch.setattr(sp.pyd, "dcmread", _fake_dcmread_factory(pixel_arrays))

    patches = sp.generate_paired_strip_patches(
        "ch2.dcm",
        "ch3.dcm",
        patch_size=300,
        patch_stride=100,
        patch_start=(0, 0),
        substrip_size=1000,
        register=False,
        patch_processor=None,
    )

    expected_coords = {(y, x) for y in range(0, 1701, 100) for x in range(0, 701, 100)}
    assert _patch_coords(patches) == expected_coords


def test_generate_paired_strip_patches_covers_grid_for_long_strip(monkeypatch):
    pixel_arrays = {
        "ch2.dcm": np.zeros((6000, 1000), dtype=np.uint16),
        "ch3.dcm": np.ones((6000, 1000), dtype=np.uint16),
    }
    monkeypatch.setattr(sp.pyd, "dcmread", _fake_dcmread_factory(pixel_arrays))

    patches = sp.generate_paired_strip_patches(
        "ch2.dcm",
        "ch3.dcm",
        patch_size=300,
        patch_stride=300,
        patch_start=(0, 0),
        substrip_size=1000,
        register=False,
        patch_processor=None,
    )

    y_coords = sorted({y for y, _ in _patch_coords(patches)})
    assert 900 in y_coords
    assert 1800 in y_coords
    assert y_coords == list(range(0, 5701, 300))
