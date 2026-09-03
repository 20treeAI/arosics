#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the SSIM validation of a single tie point in arosics.CoReg.

These drive COREG._validate_ssim_improvement directly against stub windows. The behaviour under test
depends on a specific image pair's pixel grid and is not reachable from the current test fixtures.
"""

import sys

import numpy as np
import pytest

from arosics import COREG, Tie_Point_Grid
from arosics.Tie_Point_Grid import UnexpectedTiePointError

# Production windows are 1024 px; these are small enough to keep the SSIM cheap and still exercise
# every branch, since skimage's structural_similarity needs only a 7 px side.
WINDOW = 64


class _StubWindow:
    """The parts of a GeoArray window that _validate_ssim_improvement touches."""

    def __init__(self, array, nodata=0, imID='ref'):
        self.arr = array
        self.nodata = nodata
        self.imID = imID
        self.prj = 'EPSG:32631'

    @property
    def shape(self):
        return self.arr.shape

    def __getitem__(self, item):
        return self.arr[item]

    def clip_to_poly(self, poly):
        """Drop the last row and column, standing in for the shrink GeoArray.clip_to_poly performs.

        `poly` is ignored; only the shape change matters here.
        """
        self.arr = self.arr[:-1, :-1]


class _StubBox:
    def __init__(self):
        self.mapPoly = type('_Poly', (), {'bounds': (0.0, 0.0, 1.0, 1.0)})()
        self.wp = (0.0, 0.0)

    def buffer_imXY(self, x, y):
        pass


class _StubFullImage:
    """Stands in for self.ref / self.shift, which _validate_ssim_improvement only re-reads through."""

    band4match = 0

    def __init__(self, array):
        self.array = array

    def get_mapPos(self, bounds, prj, rspAlg=None, band2get=None):
        return self.array, None, prj


def _window(size, nodata_pixels):
    """A window of `size` x `size`, containing nodata (0) pixels only when asked for."""
    array = np.arange(size * size, dtype=np.float32).reshape(size, size) % 200
    array += 0.0 if nodata_pixels else 1.0
    return array


def _coreg(match_size, other_size, deshifted_size, nodata_pixels=True):
    """A COREG whose windows have the given shapes, with __init__ bypassed.

    __init__ needs two real rasters; the attributes _validate_ssim_improvement reads are set here
    instead.
    """
    CR = COREG.__new__(COREG)
    CR.success = True
    CR.q = True
    CR._ssim_improved = None
    CR.ssim_orig = None
    CR.ssim_deshifted = None
    CR.x_shift_px, CR.y_shift_px = 1.4, 1.4
    CR.matchWin = _StubWindow(_window(match_size, nodata_pixels), imID='ref')
    CR.otherWin = _StubWindow(_window(other_size, nodata_pixels), imID='shift')
    CR.matchBox = _StubBox()
    # neither the re-read match window nor the de-shifted stub responds to buffer_imXY, so once the
    # shapes disagree they stay unequalizable
    CR.ref = _StubFullImage(_window(match_size, nodata_pixels))
    CR.shift = _StubFullImage(_window(match_size, nodata_pixels))
    CR._get_deshifted_otherWin = lambda: _StubWindow(_window(deshifted_size, nodata_pixels), imID='shift')
    return CR


@pytest.mark.parametrize('shape_a, shape_b, expected', [
    ((1024, 1024), (1024, 1024), (1024, 1024)),  # already equal and even: unchanged
    ((1024, 1024), (1023, 1023), (1022, 1022)),  # other window smaller
    ((1023, 1023), (1024, 1024), (1022, 1022)),  # match window smaller
    ((1025, 1023), (1023, 1025), (1022, 1022)),  # smaller on a different axis each
    ((512, 999), (512, 999), (512, 998)),        # odd axis rounded down
])
def test_common_even_shape(shape_a, shape_b, expected):
    """Both windows must end up the same even shape, whichever of them was larger."""
    assert COREG._common_even_shape(shape_a, shape_b) == expected
    assert COREG._common_even_shape(shape_b, shape_a) == expected


@pytest.mark.parametrize('nodata_pixels', [True, False], ids=['with_nodata', 'without_nodata'])
def test_equal_window_shapes_yield_a_similarity(nodata_pixels):
    """Identical windows score a perfect similarity, masked or not."""
    CR = _coreg(match_size=WINDOW, other_size=WINDOW, deshifted_size=WINDOW,
                nodata_pixels=nodata_pixels)

    ssim_orig, ssim_deshifted = CR._validate_ssim_improvement()

    assert ssim_orig == pytest.approx(1.0)
    assert ssim_deshifted == pytest.approx(1.0)
    assert CR.ssim_improved


@pytest.mark.parametrize('nodata_pixels', [True, False], ids=['with_nodata', 'without_nodata'])
def test_unequal_window_shapes_are_rejected(nodata_pixels):
    """Windows that differ by a pixel are not silently compared over their overlap.

    Without the shape check, numpy raises when both masked arrays carry a real mask and skimage raises
    when neither does, so the same defect surfaces under two unrelated messages.
    """
    CR = _coreg(match_size=WINDOW - 1, other_size=WINDOW, deshifted_size=WINDOW - 1,
                nodata_pixels=nodata_pixels)

    with pytest.raises(RuntimeError, match='They must be equal'):
        CR._validate_ssim_improvement()


def test_failed_shape_equalization_records_its_outcome():
    """Bailing out of the de-shifted SSIM still leaves ssim_improved decided."""
    CR = _coreg(match_size=WINDOW, other_size=WINDOW, deshifted_size=WINDOW - 1)

    with pytest.warns(UserWarning, match='could not be equalized'):
        CR._validate_ssim_improvement()

    assert CR.ssim_deshifted == 0
    assert CR._ssim_improved == False  # noqa: E712  - numpy bool, so `is False` would not hold
    assert CR.matchWin.shape != CR.otherWin.shape, 'expected the bail-out path to have clipped matchWin'


def test_ssim_improved_does_not_recompute_after_failed_equalization():
    """Reading the property after the bail-out must not re-enter the method.

    Re-entering would compare a clipped matchWin against an unclipped otherWin.
    """
    CR = _coreg(match_size=WINDOW, other_size=WINDOW, deshifted_size=WINDOW - 1)

    with pytest.warns(UserWarning, match='could not be equalized'):
        CR._validate_ssim_improvement()

    def fail_on_second_call():
        raise AssertionError('ssim_improved recomputed an already recorded result')

    CR._validate_ssim_improvement = fail_on_second_call

    assert not CR.ssim_improved


def _stub_coreg_module_attr(monkeypatch, replacement):
    """Replace COREG as Tie_Point_Grid sees it.

    arosics/__init__.py rebinds the name `arosics.Tie_Point_Grid` from the module to the class, so the
    module has to be reached through sys.modules.
    """
    monkeypatch.setattr(sys.modules['arosics.Tie_Point_Grid'], 'COREG', replacement)


class _StubCoreg:
    """A COREG that matched successfully, exposing only the attributes the result row reads."""

    tracked_errors = []
    matchBox = None
    success = True
    ref_any_nodata = True
    x_shift_px = y_shift_px = x_shift_map = y_shift_map = 0.5
    vec_length_map = vec_angle_deg = ssim_orig = ssim_deshifted = 0.5
    ssim_improved = True
    shift_reliability = 90.0

    def __init__(self, *args, **kwargs):
        pass

    def calculate_spatial_shifts(self):
        return 'success'


def test_matched_tie_point_row_matches_the_column_list(monkeypatch):
    """The success path must stay aligned with _RESULT_COLUMNS, which names the row's columns."""
    _stub_coreg_module_attr(monkeypatch, _StubCoreg)

    row = Tie_Point_Grid._get_spatial_shifts(None, None, 7)

    assert len(row) == len(Tie_Point_Grid._RESULT_COLUMNS)
    assert row[Tie_Point_Grid._RESULT_COLUMNS.index('POINT_ID')] == 7
    assert row[Tie_Point_Grid._RESULT_COLUMNS.index('RELIABILITY')] == 90.0


def test_raising_tie_point_is_reported_as_unmatched(monkeypatch):
    """A tie point that raises comes back as an unmatched row rather than propagating."""
    def boom(*args, **kwargs):
        raise ValueError('operands could not be broadcast together')

    _stub_coreg_module_attr(monkeypatch, boom)

    with pytest.warns(UserWarning, match='recorded without a match'):
        row = Tie_Point_Grid._get_spatial_shifts(None, None, 42)

    assert len(row) == len(Tie_Point_Grid._RESULT_COLUMNS)
    assert row[Tie_Point_Grid._RESULT_COLUMNS.index('POINT_ID')] == 42
    assert all(value is None for value in row[1:-1])

    error = row[Tie_Point_Grid._RESULT_COLUMNS.index('LAST_ERR')]
    assert isinstance(error, UnexpectedTiePointError)
    assert 'operands could not be broadcast together' in str(error)
    assert '_get_spatial_shifts' in str(error), 'expected the traceback to be preserved'


def test_recorded_error_survives_a_process_boundary():
    """A result row is pickled back from a worker, so LAST_ERR has to round-trip."""
    import pickle

    def boom(*args, **kwargs):
        raise ValueError('operands could not be broadcast together')

    original = sys.modules['arosics.Tie_Point_Grid'].COREG
    sys.modules['arosics.Tie_Point_Grid'].COREG = boom
    try:
        with pytest.warns(UserWarning):
            row = Tie_Point_Grid._get_spatial_shifts(None, None, 42)
    finally:
        sys.modules['arosics.Tie_Point_Grid'].COREG = original

    restored = pickle.loads(pickle.dumps(row))

    assert isinstance(restored[-1], UnexpectedTiePointError)
    assert str(restored[-1]) == str(row[-1])
