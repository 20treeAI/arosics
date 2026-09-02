#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the SSIM validation of a single tie point in arosics.CoReg.

These drive COREG._validate_ssim_improvement directly against stub windows, because the behaviour under
test only shows up when the de-shifted window cannot be shape-matched to the match window - a condition
that depends on the pixel grid of a specific image pair and cannot be provoked from the test fixtures.
"""

import numpy as np
import pytest

from arosics import COREG, Tie_Point_Grid


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
        """Shrink the window, as GeoArray.clip_to_poly does when the box no longer covers the array."""
        self.arr = self.arr[:-1, :-1]


class _StubBox:
    def __init__(self):
        self.mapPoly = type('_Poly', (), {'bounds': (0.0, 0.0, 1.0, 1.0)})()
        self.buffered = False

    def buffer_imXY(self, x, y):
        self.buffered = True


class _StubFullImage:
    """Stands in for self.ref / self.shift, which are only used to re-read the match window."""

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

    __init__ needs two real rasters and computes a matching window from them; every attribute it would
    set that _validate_ssim_improvement reads is set here instead.
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
    # the re-read match window keeps its original size, so the shapes stay unequalizable
    CR.ref = _StubFullImage(_window(match_size, nodata_pixels))
    CR.shift = _StubFullImage(_window(match_size, nodata_pixels))
    CR._get_deshifted_otherWin = lambda: _StubWindow(_window(deshifted_size, nodata_pixels), imID='shift')
    return CR


@pytest.mark.parametrize('nodata_pixels', [True, False], ids=['with_nodata', 'without_nodata'])
def test_unequal_window_shapes_do_not_raise(nodata_pixels):
    """Windows that differ by a pixel must still yield an SSIM.

    Unequal shapes used to raise out of the joblib worker and abort the co-registration of the whole
    image: a ValueError from numpy when both masks were real, and one from skimage when neither was.
    """
    CR = _coreg(match_size=1023, other_size=1024, deshifted_size=1023, nodata_pixels=nodata_pixels)

    ssim_orig, ssim_deshifted = CR._validate_ssim_improvement()

    assert isinstance(ssim_orig, float)
    assert isinstance(ssim_deshifted, float)


def test_failed_shape_equalization_records_its_outcome():
    """Bailing out of the de-shifted SSIM must still leave ssim_improved decided.

    The bail-out path clips self.matchWin, so leaving _ssim_improved unset let the ssim_improved property
    run the whole method a second time - against a match window that no longer matched self.otherWin.
    """
    CR = _coreg(match_size=1024, other_size=1024, deshifted_size=1023)

    with pytest.warns(UserWarning, match='could not be equalized'):
        CR._validate_ssim_improvement()

    assert CR.ssim_deshifted == 0
    assert CR._ssim_improved is not None
    assert CR.matchWin.shape != CR.otherWin.shape, 'expected the bail-out path to have clipped matchWin'


def test_ssim_improved_does_not_recompute_after_failed_equalization():
    CR = _coreg(match_size=1024, other_size=1024, deshifted_size=1023)

    with pytest.warns(UserWarning, match='could not be equalized'):
        CR._validate_ssim_improvement()

    def fail_on_second_call():
        raise AssertionError('ssim_improved recomputed an already recorded result')

    CR._validate_ssim_improvement = fail_on_second_call

    assert CR.ssim_improved is False


def test_raising_tie_point_is_reported_as_unmatched(monkeypatch):
    """One unmatchable tie point must not abort the grid, and with it the whole image."""
    def boom(*args, **kwargs):
        raise ValueError('operands could not be broadcast together')

    monkeypatch.setattr('arosics.Tie_Point_Grid.COREG', boom)

    with pytest.warns(UserWarning, match='is dropped from the grid'):
        row = Tie_Point_Grid._get_spatial_shifts(None, None, 42)

    assert len(row) == len(Tie_Point_Grid._RESULT_COLUMNS)
    assert row[Tie_Point_Grid._RESULT_COLUMNS.index('POINT_ID')] == 42
    assert isinstance(row[Tie_Point_Grid._RESULT_COLUMNS.index('LAST_ERR')], ValueError)
    assert all(value is None for value in row[1:-1])
