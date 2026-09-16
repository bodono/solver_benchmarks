These are authored regression problems, not downloaded benchmark data.

- `tiny.emps`: four rows (objective plus L/G/E), six columns, 30 shared
  numbers, RHS and ranges, and each of UP/LO/FX/FR/MI/PL bounds. Encodings
  exercise signed integers, decimal exponents, shared-number indices above
  22 and a mantissa stored in two groups. Expected model arrays are in
  `test_emps.py`.
- `blanksets.emps`: the same problem with valid empty RHS, RANGES and BOUNDS
  set names; the output uses the corresponding section name.
- `checksums.emps`: 143 rows and one column, crossing two 71-record checksum
  boundaries and using a multi-character row index.

Both fixtures and the Python output were independently checked against David
M. Gay's NETLIB `emps.c` decoder. No upstream source or benchmark data is
included in these fixtures.
