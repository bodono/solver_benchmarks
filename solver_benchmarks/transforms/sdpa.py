"""SDPA-S (sparse) format parser.

SDPA-S is the standard text format for SDP benchmark suites including
SDPLIB and Mittelmann's SDP test set. Each ``.dat-s`` file describes
the primal SDP::

    minimize    C • X
    subject to  A_k • X = b_k,   k = 1..m
                X is block-diagonal, each block PSD or NN (diagonal).

We convert it to the CONE-form dual problem this codebase uses::

    minimize    q' x
    subject to  A x + s = b,  s in K

where x is the SDP dual ``y``, ``q = -b_sdpa`` (we minimize the dual,
hence the sign flip), columns of ``A`` are the vectorized ``A_k``,
``b = vec(C)``, and ``K`` is the product of the SDP's PSD / NN blocks
in the canonical layout (``s``: list of PSD orders; ``l``: count of
diagonal/NN-block rows). PSD entries are vectorized in column-major
lower order with √2 scaling on off-diagonals — matching the layout
in `transforms.psd`.

File format reference (SDPA-S):

    "* comments (start with * or ")
    m                          (number of equality constraints)
    nblocks                    (number of blocks in X)
    blocksizes                 (positive: PSD block of that order;
                                negative: |size| diagonal block)
    b_1 b_2 ... b_m            (right-hand side)
    matno blockno i j value    (one line per non-zero of A_matno's
                                blockno-th block; matno=0 → C; i ≤ j;
                                only the lower triangle is given)
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class SDPABlock:
    """One block of an SDPA matrix.

    ``order`` is the matrix dimension. ``is_psd`` says whether this
    is a PSD block (size > 0 in the file) or a diagonal/NN block
    (size < 0 in the file, |size| diagonal entries).
    """

    order: int
    is_psd: bool


@dataclass
class SDPAProblem:
    """Parsed SDPA-S contents in primal form.

    ``c_blocks`` and ``a_blocks`` are lists keyed first by constraint
    (``a_blocks[k][b]`` is constraint ``k``'s entries for block ``b``)
    and then by block. Entries are stored as ``(i, j, value)`` triples
    with ``i ≤ j`` (lower triangle, BLAS-style); every matrix is
    symmetric.
    """

    m: int
    blocks: list[SDPABlock]
    b: np.ndarray
    c_blocks: list[list[tuple[int, int, float]]]
    a_blocks: list[list[list[tuple[int, int, float]]]]


def parse_sdpa_s(text: str) -> SDPAProblem:
    """Parse SDPA-S text into an :class:`SDPAProblem`.

    Tolerates the format's flexibility: comments may start with ``*``
    or ``"`` and appear anywhere; numeric tokens may be separated by
    whitespace or commas; block sizes may be on one line or several.
    """
    lines = _strip_comments(text)
    tokens = _tokenize_lines(lines)
    if not tokens:
        raise ValueError("SDPA-S file is empty after comment stripping.")

    cursor = 0
    m, cursor = _take_int(tokens, cursor)
    nblocks, cursor = _take_int(tokens, cursor)
    raw_block_sizes: list[int] = []
    for _ in range(nblocks):
        size, cursor = _take_int(tokens, cursor)
        raw_block_sizes.append(size)
    blocks = [
        SDPABlock(order=abs(size), is_psd=size > 0)
        for size in raw_block_sizes
    ]

    b = np.zeros(m, dtype=float)
    for k in range(m):
        value, cursor = _take_float(tokens, cursor)
        b[k] = value

    # Initialize empty entry lists for C (matno=0) and each A_k.
    c_blocks: list[list[tuple[int, int, float]]] = [[] for _ in blocks]
    a_blocks: list[list[list[tuple[int, int, float]]]] = [
        [[] for _ in blocks] for _ in range(m)
    ]

    while cursor + 5 <= len(tokens):
        matno, cursor = _take_int(tokens, cursor)
        blockno, cursor = _take_int(tokens, cursor)
        i, cursor = _take_int(tokens, cursor)
        j, cursor = _take_int(tokens, cursor)
        value, cursor = _take_float(tokens, cursor)
        if matno < 0 or matno > m:
            raise ValueError(
                f"SDPA-S matno {matno} out of range [0, {m}]"
            )
        if blockno < 1 or blockno > len(blocks):
            raise ValueError(
                f"SDPA-S blockno {blockno} out of range [1, {len(blocks)}]"
            )
        # 1-indexed in the file; convert to 0-indexed entries.
        i0 = i - 1
        j0 = j - 1
        if i0 > j0:
            i0, j0 = j0, i0
        if matno == 0:
            c_blocks[blockno - 1].append((i0, j0, float(value)))
        else:
            a_blocks[matno - 1][blockno - 1].append(
                (i0, j0, float(value))
            )
    return SDPAProblem(m=m, blocks=blocks, b=b, c_blocks=c_blocks, a_blocks=a_blocks)


def parse_sdpa_s_file(path: Path) -> SDPAProblem:
    """Read an SDPA-S file (plain or gzipped) and parse it."""
    if path.suffix == ".gz" or path.name.endswith(".dat-s.gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    return parse_sdpa_s(text)


def sdpa_to_cone_problem(problem: SDPAProblem) -> dict:
    """Convert a parsed primal SDP into the CONE-form dual problem.

    Layout of the cone-form rows (matching the schema's iteration
    order over cone keys ``z`` < ``l`` < ``q`` < ``s``):

    - ``l`` rows: one row per diagonal entry of every NN (negative-
      size) block. We do not currently emit equality (``z``) blocks
      because SDPA-S does not have a notion of fixed-equality
      variables in its primal representation; ``X`` is constrained to
      be PSD-block-diagonal.
    - ``s`` blocks: one block per PSD block, vectorized in canonical
      column-major lower order with ``√2`` scaling on off-diagonals.

    Returns a dict matching the canonical CONE problem schema:
    ``{"P", "q", "A", "b", "cone", ...}``.
    """
    # Compute total row counts per cone kind.
    l_rows = sum(blk.order for blk in problem.blocks if not blk.is_psd)
    psd_orders = [blk.order for blk in problem.blocks if blk.is_psd]
    psd_triangle_rows = sum(order * (order + 1) // 2 for order in psd_orders)
    total_rows = l_rows + psd_triangle_rows

    # Offsets into the row-stacked output for each block.
    block_offsets = _block_offsets(problem.blocks)

    # Build A and b. b = vec(C) in the same row layout.
    b_dense = np.zeros(total_rows, dtype=float)
    for blk_idx, entries in enumerate(problem.c_blocks):
        block = problem.blocks[blk_idx]
        offset = block_offsets[blk_idx]
        if block.is_psd:
            _write_psd_triangle_into_vec(b_dense, offset, block.order, entries)
        else:
            _write_diag_into_vec(b_dense, offset, entries)

    # Each constraint k contributes one COLUMN of the cone-form A:
    # column k of A is the vec of A_k in the same row layout. (SDPA's
    # primal "A_k • X = b_k" becomes the dual cone constraint with
    # rows of the dual A indexed by primal X-vec entries and columns
    # by dual y_k.)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for k, k_blocks in enumerate(problem.a_blocks):
        for blk_idx, entries in enumerate(k_blocks):
            block = problem.blocks[blk_idx]
            offset = block_offsets[blk_idx]
            if block.is_psd:
                for i, j, value in entries:
                    if i == j:
                        row = offset + _psd_triangle_index(block.order, i, j)
                        rows.append(row)
                        cols.append(k)
                        data.append(float(value))
                    else:
                        row = offset + _psd_triangle_index(block.order, i, j)
                        rows.append(row)
                        cols.append(k)
                        # Off-diagonal entry (i ≠ j) in the canonical
                        # √2-scaled vec corresponds to ``√2 * X[i,j]``;
                        # the symmetric SDPA inner product
                        # ``A_k • X = sum_ij A_k[i,j] X[i,j]``
                        # has 2 * A_k[i,j] X[i,j] for the off-diagonal
                        # entry pair. So vec(A_k) has ``√2 * A_k[i,j]``
                        # to make the canonical inner product match.
                        data.append(float(value) * float(np.sqrt(2.0)))
            else:
                for i, j, value in entries:
                    if i != j:
                        # Non-PSD blocks should be diagonal; off-diag
                        # entries are not allowed in a NN/diagonal block.
                        raise ValueError(
                            "SDPA-S diagonal block has off-diagonal entry "
                            f"(block {blk_idx + 1}, i={i+1}, j={j+1})."
                        )
                    rows.append(offset + i)
                    cols.append(k)
                    data.append(float(value))

    a_matrix = sp.csc_matrix(
        (data, (rows, cols)),
        shape=(total_rows, problem.m),
    )
    cone: dict = {}
    if l_rows:
        cone["l"] = int(l_rows)
    if psd_orders:
        cone["s"] = [int(order) for order in psd_orders]

    return {
        "P": None,
        "q": -problem.b.copy(),  # dual objective: max b'y → min -b'y
        "r": 0.0,
        "A": a_matrix,
        "b": b_dense,
        "n": int(problem.m),
        "m": int(total_rows),
        "cone": cone,
        "obj_type": "min",
    }


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _strip_comments(text: str) -> str:
    """Drop SDPA-style comment lines.

    Comments begin with ``*`` or ``"`` and run to the end of the line.
    The standard recommends comments only appear at the top, but real
    files in the wild sometimes interleave them; treat both leading-
    line and inline comments as stripped from that character to EOL.
    """
    out_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[0] in ('"', "*"):
            continue
        # Strip inline comments after a `*` or `"` if present.
        for marker in ("*", '"'):
            idx = line.find(marker)
            if idx >= 0:
                line = line[:idx]
        out_lines.append(line)
    return "\n".join(out_lines)


_TOKEN_SPLIT = re.compile(r"[\s,;{}()]+")


def _tokenize_lines(text: str) -> list[str]:
    """Split the cleaned text into numeric tokens.

    SDPA files use a permissive separator set (whitespace, commas,
    parentheses, braces). The first non-empty string after splitting
    on those separators is the next token.
    """
    return [tok for tok in _TOKEN_SPLIT.split(text) if tok]


def _take_int(tokens: list[str], cursor: int) -> tuple[int, int]:
    if cursor >= len(tokens):
        raise ValueError("Unexpected end of SDPA-S file (expected integer).")
    return int(tokens[cursor]), cursor + 1


def _take_float(tokens: list[str], cursor: int) -> tuple[float, int]:
    if cursor >= len(tokens):
        raise ValueError("Unexpected end of SDPA-S file (expected float).")
    return float(tokens[cursor]), cursor + 1


def _block_offsets(blocks: list[SDPABlock]) -> list[int]:
    """Compute the row offset of each block in the canonical row-
    stacked output. Diagonal/NN blocks come before PSD blocks,
    matching the cone-key ordering ``l`` < ``s``.
    """
    offsets = [0] * len(blocks)
    cursor = 0
    # First pass: NN blocks.
    for idx, blk in enumerate(blocks):
        if not blk.is_psd:
            offsets[idx] = cursor
            cursor += blk.order
    # Second pass: PSD blocks (use canonical n*(n+1)/2 entry count).
    for idx, blk in enumerate(blocks):
        if blk.is_psd:
            offsets[idx] = cursor
            cursor += blk.order * (blk.order + 1) // 2
    return offsets


def _psd_triangle_index(order: int, i: int, j: int) -> int:
    """Index of ``X[i, j]`` in a column-major lower-triangle vec of an
    ``order x order`` PSD block. Assumes ``i >= j``.
    """
    if i < j:
        i, j = j, i
    # Column j has order - j entries (from row j to row order-1),
    # but we want the running offset. Sum over preceding columns.
    return j * order - j * (j - 1) // 2 + (i - j)


def _write_psd_triangle_into_vec(
    out: np.ndarray,
    offset: int,
    order: int,
    entries: list[tuple[int, int, float]],
) -> None:
    """Write entries of one PSD block into the canonical vec at
    ``out[offset:offset + order*(order+1)//2]``. Off-diagonal entries
    get the ``√2`` scaling that the canonical inner product expects.
    """
    sqrt2 = float(np.sqrt(2.0))
    for i, j, value in entries:
        idx = offset + _psd_triangle_index(order, i, j)
        if i == j:
            out[idx] += float(value)
        else:
            out[idx] += float(value) * sqrt2


def _write_diag_into_vec(
    out: np.ndarray,
    offset: int,
    entries: list[tuple[int, int, float]],
) -> None:
    """Write diagonal-block entries into the row-stacked vec."""
    for i, j, value in entries:
        if i != j:
            raise ValueError(
                "SDPA-S diagonal block has off-diagonal entry "
                f"(i={i+1}, j={j+1})."
            )
        out[offset + i] += float(value)
