"""Streaming reader for NETLIB's compressed MPS (EMPS) interchange format.

This is an independent Python implementation of the file encoding, with no
compiler or external executable required. Format references and independent
validation tools are David M. Gay's NETLIB expanders:
https://www.netlib.org/lp/data/emps.c and https://www.netlib.org/lp/data/emps.f.
Neither source is bundled or compiled by this package.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TextIO

# The format uses printable ASCII except space, colon, backslash and DEL.
_ALPHABET = ''.join(chr(i) for i in range(33, 127) if chr(i) not in ':\\')
_DIGITS = {character: value for value, character in enumerate(_ALPHABET)}
_BOUND_TYPES = ("UP", "LO", "FX", "FR", "MI", "PL")


def is_emps_header(lines: list[str]) -> bool:
    """Recognize the two dimension records following an EMPS NAME record."""
    records = [line.strip() for line in lines if line.strip() and not line.startswith('*')]
    if len(records) < 3 or records[0].split()[0] != "NAME":
        return False
    fields = [records[1].split(), records[2].split()]
    return [len(row) for row in fields] == [8, 3] and all(
        token.isdecimal() for row in fields for token in row
    )


class _Records:
    """Read physical records and validate each block's per-line checksums."""

    def __init__(self, source: TextIO):
        self.source = source
        self.checksums = ""
        self.line_number = 0

    def raw(self) -> str:
        line = self.source.readline()
        self.line_number += 1
        if not line:
            raise ValueError("Truncated EMPS file")
        return line.rstrip('\r\n')

    def verify(self) -> None:
        if self.raw() != " " + self.checksums:
            raise ValueError(f"EMPS checksum mismatch at line {self.line_number}")
        self.checksums = ""

    def next(self) -> str:
        line = self.raw()
        if line.startswith(':'):
            # Extension records can change the model (e.g. integer markers).
            raise ValueError("EMPS extension records are not supported")
        checksum = 0
        for character in line:
            checksum = (checksum >> 1) + (checksum & 1) * 16384 + _DIGITS.get(character, 92)
        self.checksums += _ALPHABET[checksum % 92]
        if len(self.checksums) == 71:
            self.verify()
        return line


class _Values:
    """Decode a section's variable-length index and decimal-value stream."""

    def __init__(self, records: _Records, numbers: list[str]):
        self.records = records
        self.numbers = numbers
        self.line = ""
        self.position = 0

    def record(self) -> None:
        if self.position != len(self.line):
            raise ValueError("Unexpected trailing EMPS data")
        self.line = self.records.next()
        self.position = 0
        if not self.line:
            raise ValueError("Empty EMPS data record")

    def digit(self) -> int:
        # Encoded values may start on the next record, but do not span records.
        if self.position == len(self.line):
            raise ValueError("Truncated EMPS encoded value")
        character = self.line[self.position]
        self.position += 1
        try:
            return _DIGITS[character]
        except KeyError:
            raise ValueError(f"Invalid EMPS encoded character: {character!r}") from None

    def start(self) -> None:
        if self.position == len(self.line):
            self.record()

    def index(self) -> int:
        self.start()
        first = self.digit()
        if first >= 46:
            raise ValueError("Invalid EMPS index prefix")
        if first >= 23:
            return first - 23
        return self.integer_tail(first)

    def integer_tail(self, value: int) -> int:
        while True:
            digit = self.digit()
            value = value * 46 + digit % 46
            if digit >= 46:
                return value

    def value(self) -> str:
        self.start()
        prefix = self.digit()
        if prefix < 46:
            self.position -= 1
            index = self.index()
            if not 1 <= index <= len(self.numbers):
                raise ValueError(f"Invalid EMPS number-table index: {index}")
            return self.numbers[index - 1]
        negative, kind = divmod(prefix - 46, 23)
        sign = "-" if negative else ""
        if kind >= 11:
            first = kind - 11
            number = first - 6 if first >= 6 else self.integer_tail(first)
            return sign + str(number)
        exponent = self.digit() - 50
        mantissa = self.digit()
        leading = None
        for _ in range(kind):
            if mantissa >= 100_000_000:
                if leading is not None:
                    raise ValueError("Invalid EMPS decimal mantissa")
                leading, mantissa = mantissa, self.digit()
            else:
                mantissa = mantissa * 92 + self.digit()
        digits = str(mantissa)
        if leading is not None:
            if not digits.startswith('1'):
                raise ValueError("Invalid EMPS decimal continuation")
            digits = str(leading) + digits[1:]
        return f"{sign}{digits}e{exponent}"

    def name(self, default: str = "") -> str:
        value = self.line[self.position:]
        self.position = len(self.line)
        return _name(value.rstrip(' ') or default)

    def finish(self) -> None:
        if self.position != len(self.line):
            raise ValueError("Unexpected trailing EMPS section data")


def _name(value: str) -> str:
    # MPS permits blanks within its eight-character fixed-width names. Use
    # underscores so the generated free-format records remain unambiguous.
    value = value.rstrip(' ')
    if not value or len(value) > 8 or any(ord(c) < 32 for c in value):
        raise ValueError(f"Invalid EMPS row/column name: {value!r}")
    return value.replace(' ', '_')


def _names_unique(names: list[str]) -> None:
    if len(set(names)) != len(names):
        raise ValueError("Duplicate EMPS names after replacing embedded blanks")


def _entries(
    records: _Records,
    numbers: list[str],
    count: int,
    names: list[str],
    *,
    bounds: bool = False,
    default_name: str = "",
) -> Iterator[tuple[str, str, str, str]]:
    values = _Values(records, numbers)
    current = ""
    for _ in range(count):
        index = values.index()
        while index == 0:
            current = values.name(default_name)
            if not bounds:
                # Column/set names are yielded separately from their entries.
                yield current, "", "", ""
            index = values.index()
        if not current:
            raise ValueError("EMPS entries precede their column/set name")
        kind = ""
        if bounds:
            if not 1 <= index <= len(_BOUND_TYPES):
                raise ValueError(f"Invalid EMPS bound type: {index}")
            kind = _BOUND_TYPES[index - 1]
            index = values.index()
        if not 1 <= index <= len(names):
            raise ValueError(f"EMPS row/column index out of range: {index}")
        value = "" if kind in {"FR", "MI", "PL"} else values.value()
        yield current, names[index - 1], value, kind
    values.finish()


def decode_emps(source: TextIO, target: TextIO) -> None:
    """Write one EMPS problem as standard MPS, rejecting corrupt input.

    Callers must write to a temporary file: validation completes only after
    the final checksum is consumed. Memory usage is proportional to the row,
    column and shared-number tables, rather than the number of nonzeros.
    """
    records = _Records(source)
    line = records.raw()
    while not line.strip() or line.startswith('*'):
        line = records.raw()
    if not line.startswith("NAME "):
        raise ValueError("EMPS file has no NAME record")
    header = [line, records.raw(), records.raw()]
    if not is_emps_header(header):
        raise ValueError("Invalid EMPS dimension records")
    nrows, ncols, _, nonzeros, nrhs, rhsnz, nranges, rangenz = map(int, header[1].split())
    nbounds, boundsnz, nvalues = map(int, header[2].split())
    if nrows == 0 or ncols == 0:
        raise ValueError("EMPS problem must contain rows and columns")
    target.write(line + "\n")

    numbers: list[str] = []
    values = _Values(records, [])
    for _ in range(nvalues):
        numbers.append(values.value())
    values.finish()

    target.write("ROWS\n")
    rows = []
    for _ in range(nrows):
        row = records.next()
        if not row or row[0] not in "NLEG":
            raise ValueError(f"Invalid EMPS row type: {row!r}")
        name = _name(row[1:])
        rows.append(name)
        target.write(f" {row[0]}  {name}\n")
    _names_unique(rows)

    columns: list[str] = []
    for section, count, set_count in (
        ("COLUMNS", nonzeros, ncols),
        ("RHS", rhsnz, nrhs),
        ("RANGES", rangenz, nranges),
    ):
        if section != "RANGES" or count:
            target.write(section + "\n")
        sets = []
        for name, row, value, _ in _entries(
            records, numbers, count, rows,
            default_name="" if section == "COLUMNS" else section,
        ):
            if not row:
                sets.append(name)
            else:
                target.write(f"    {name:<8}  {row:<8}  {value}\n")
        if len(sets) != set_count:
            raise ValueError(f"EMPS {section} count does not match its header")
        _names_unique(sets)
        if section == "COLUMNS":
            columns = sets

    if boundsnz:
        target.write("BOUNDS\n")
    bound_sets = set()
    for name, column, value, kind in _entries(
        records, numbers, boundsnz, columns, bounds=True, default_name="BOUNDS"
    ):
        bound_sets.add(name)
        target.write(f" {kind} {name:<8}  {column:<8}  {value}\n")
    if len(bound_sets) != nbounds:
        raise ValueError("EMPS BOUNDS count does not match its header")
    if records.checksums:
        records.verify()
    if any(line.strip() for line in source):
        raise ValueError("Unexpected data after the EMPS problem")
    target.write("ENDATA\n")
