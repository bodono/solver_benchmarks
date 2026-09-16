"""EMPS regression fixtures describe authored LPs, independent of downloads."""

import io
from decimal import Decimal
from pathlib import Path

import highspy
import numpy as np
import pytest

from solver_benchmarks.transforms.emps import _Records, _Values, decode_emps, is_emps_header

FIXTURES = Path(__file__).parent / "fixtures" / "emps"


@pytest.mark.parametrize("fixture", ["tiny.emps", "blanksets.emps"])
def test_decode_rows_columns_ranges_and_all_six_bound_types(tmp_path, fixture):
    path = tmp_path / "tiny.mps"
    with (FIXTURES / fixture).open() as source, path.open("w") as target:
        decode_emps(source, target)
    reader = highspy.Highs()
    reader.setOptionValue("output_flag", False)
    assert reader.readModel(str(path)) == highspy.HighsStatus.kOk
    lp = reader.getLp()
    assert (lp.num_row_, lp.num_col_) == (3, 6)
    np.testing.assert_array_equal(lp.col_cost_, [1, 1000.00001123, 0, 0, 0, 0])
    np.testing.assert_array_equal(lp.col_lower_, [0, -2, 1, -np.inf, -np.inf, 0])
    np.testing.assert_array_equal(lp.col_upper_, [8, np.inf, 1, np.inf, np.inf, np.inf])
    np.testing.assert_array_equal(lp.row_lower_, [8, -1, -1])
    np.testing.assert_array_equal(lp.row_upper_, [10, 2, 3])
    np.testing.assert_array_equal(lp.a_matrix_.start_, [0, 3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(lp.a_matrix_.index_, [0, 1, 2, 0, 0, 1, 2, 0])
    np.testing.assert_array_equal(lp.a_matrix_.value_, [2.5, -7, 46, 1, 1, 1, 1, 1])


@pytest.mark.parametrize(
    ("encoded", "expected"),
    [
        ('PS;', '2.5'),
        ('sW', '-7'),
        (']P', '46'),
        ('iR"C', '-1.25'),
        ('P@"', '1e-20'),
        ('VL"FHgS-4', '1000.00001123'),
    ],
)
def test_numeric_encodings_preserve_decimal_values(encoded, expected):
    # These tokens are also in the fixture's shared table. The unused entries
    # here cover very small and negative fractional coefficients explicitly.
    stream = _Values(_Records(io.StringIO(encoded + "\n")), [])
    assert Decimal(stream.value()) == Decimal(expected)
    stream.finish()


def test_decode_multiple_checksum_blocks_and_multicharacter_indices():
    out = io.StringIO()
    with (FIXTURES / "checksums.emps").open() as source:
        decode_emps(source, out)
    text = out.getvalue()
    assert text.count(" L  R") == 142
    assert "    X         R142      2\n" in text
    assert text.endswith("RHS\nENDATA\n")


@pytest.mark.parametrize("fixture", ["tiny.emps", "checksums.emps"])
def test_rejects_checksum_corruption(fixture):
    body = (FIXTURES / fixture).read_text()
    body = body.replace("NCOST", "NLOST")
    with pytest.raises(ValueError, match="checksum mismatch"):
        decode_emps(io.StringIO(body), io.StringIO())


@pytest.mark.parametrize("remove", [1, 4, 12])
def test_rejects_truncated_files(remove):
    lines = (FIXTURES / "tiny.emps").read_text().splitlines(keepends=True)
    with pytest.raises(ValueError, match="Truncated"):
        decode_emps(io.StringIO(''.join(lines[:-remove])), io.StringIO())


def test_rejects_bad_header_count_and_trailing_data():
    body = (FIXTURES / "tiny.emps").read_text()
    with pytest.raises(ValueError, match="count does not match"):
        decode_emps(io.StringIO(body.replace("4 6 4 10", "4 7 4 10")), io.StringIO())
    with pytest.raises(ValueError, match="Unexpected data after"):
        decode_emps(io.StringIO(body + "extra\n"), io.StringIO())


def test_header_detection_distinguishes_standard_mps():
    assert is_emps_header(["NAME model", "4 6 4 10 1 3 1 3", "1 6 30"])
    assert not is_emps_header(["NAME model", "ROWS", " N COST"])
    assert not is_emps_header(["NAME model", "-4 6 4 10 1 3 1 3", "1 6 30"])
