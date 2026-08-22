"""Apply the PR #54 changes to a freshly checked-out scs-python tree.

Mirrors the aa-trust-factor branch's modifications (which target the
canonical aa.c) into scs-python's vendored copy + scs.c + glbopts.h.

Idempotent: bails out if AA_TRUST_FACTOR is already defined.
"""
from __future__ import annotations

from pathlib import Path

SCS = Path("/Users/bodonoghue/git/scs-python/scs_source")


def patch_file(path: Path, old: str, new: str, *, allow_present: bool = False) -> bool:
    txt = path.read_text()
    if new in txt and old not in txt:
        if allow_present:
            return False
        raise SystemExit(f"{path}: already patched")
    if old not in txt:
        raise SystemExit(f"{path}: anchor not found")
    path.write_text(txt.replace(old, new))
    return True


def main() -> None:
    glb = SCS / "include" / "glbopts.h"
    if "AA_TRUST_FACTOR" in glb.read_text():
        print("scs-python already patched; nothing to do.")
        return

    # 1. Add max_gamma_norm / trust_factor field to ACCEL_WORK
    aa_c = SCS / "src" / "aa.c"
    patch_file(aa_c,
        "  aa_float max_weight_norm;  /* maximum norm of AA weights */",
        """  aa_float max_weight_norm;  /* maximum norm of AA weights */
  aa_float trust_factor;     /* opt-in trust region + adaptive r */
  aa_float r_adaptive;       /* adaptive r state, used only when trust active */""")

    # 2. Replace compute_regularization
    old_cr = """static aa_float compute_regularization(AaWork *a) {
  TIME_TIC
  aa_float nrm_y = frob_from_col_norms(a->nrm_y_col, a->mem);
  aa_float nrm_a = a->type1 ? frob_from_col_norms(a->nrm_s_col, a->mem) : nrm_y;
  aa_float r = a->regularization * nrm_a * nrm_y;
  if (a->verbosity > 2) {
    scs_printf(\"iter: %i, ||A||_F %.2e, ||Y||_F %.2e, r: %.2e\\n\",
               (int)a->iter, nrm_a, nrm_y, r);
  }
  TIME_TOC
  return r;
}"""
    new_cr = """static aa_float compute_regularization(AaWork *a) {
  TIME_TIC
  aa_float r;
  if (isfinite(a->trust_factor)) {
    r = a->r_adaptive;
  } else {
    aa_float nrm_y = frob_from_col_norms(a->nrm_y_col, a->mem);
    aa_float nrm_s = frob_from_col_norms(a->nrm_s_col, a->mem);
    r = a->regularization * nrm_s * nrm_y;  /* sym */
  }
  TIME_TOC
  return r;
}

static void trust_grow(AaWork *a) {
  if (!isfinite(a->trust_factor)) return;
  a->r_adaptive *= 10.0;
  if (a->r_adaptive > 1e30) a->r_adaptive = 1e30;
}
static void trust_shrink(AaWork *a) {
  if (!isfinite(a->trust_factor)) return;
  a->r_adaptive *= 0.9;
  if (a->r_adaptive < 1e-12) a->r_adaptive = 1e-12;
}"""
    patch_file(aa_c, old_cr, new_cr)

    # 3. In-solve rejection path: trust_grow
    patch_file(aa_c,
        """    } else {
      a->n_reject_weight_cap++;
    }
    a->success = 0;
    aa_reset(a);""",
        """    } else {
      a->n_reject_weight_cap++;
    }
    trust_grow(a);
    a->success = 0;
    aa_reset(a);""")

    # 4. Trust region in solve() before f -= Dγ
    patch_file(aa_c,
        """    if (!isfinite(aa_norm)) aa_norm = -1.0;
    return (aa_norm < 0) ? aa_norm : -aa_norm;
  }

  /* f -= D γ */
  BLAS(gemv)
  (\"NoTrans\", &bdim, &blen, &neg_onef, a->D, &bdim, gamma, &one, &onef, f,
   &one);""",
        """    if (!isfinite(aa_norm)) aa_norm = -1.0;
    return (aa_norm < 0) ? aa_norm : -aa_norm;
  }

  if (isfinite(a->trust_factor)) {
    aa_float zerof = 0.0;
    aa_float d_gamma_norm;
    BLAS(gemv)
    (\"NoTrans\", &bdim, &blen, &onef, a->D, &bdim, gamma, &one, &zerof,
     a->c_aug, &one);
    d_gamma_norm = BLAS(nrm2)(&bdim, a->c_aug, &one);
    if (isfinite(d_gamma_norm) && d_gamma_norm > a->trust_factor * a->norm_g) {
      a->n_reject_weight_cap++;
      trust_grow(a);
      a->success = 0;
      aa_reset(a);
      TIME_TOC
      return -aa_norm;
    }
  }

  /* f -= D γ */
  BLAS(gemv)
  (\"NoTrans\", &bdim, &blen, &neg_onef, a->D, &bdim, gamma, &one, &onef, f,
   &one);""")

    # 5. aa_init signature
    patch_file(aa_c,
        """AaWork *aa_init(aa_int dim, aa_int mem, aa_int min_len, aa_int type1,
                aa_float regularization, aa_float relaxation,
                aa_float safeguard_factor, aa_float max_weight_norm,
                aa_int ir_max_steps, aa_int verbosity) {""",
        """AaWork *aa_init(aa_int dim, aa_int mem, aa_int min_len, aa_int type1,
                aa_float regularization, aa_float relaxation,
                aa_float safeguard_factor, aa_float max_weight_norm,
                aa_float trust_factor, aa_int ir_max_steps,
                aa_int verbosity) {""")

    # 6. Validation
    patch_file(aa_c,
        """  if (dim <= 0 || mem < 0 || !isfinite(regularization) ||
      relaxation < 0 || relaxation > 2 ||
      safeguard_factor < 0 || max_weight_norm <= 0 ||
      ir_max_steps < 0 ||
      (mem_clamped > 0 && min_len < 1)) {""",
        """  if (dim <= 0 || mem < 0 || !isfinite(regularization) ||
      relaxation < 0 || relaxation > 2 ||
      safeguard_factor < 0 ||
      isnan(max_weight_norm) || max_weight_norm <= 0 ||
      isnan(trust_factor) || trust_factor <= 0 ||
      ir_max_steps < 0 ||
      (mem_clamped > 0 && min_len < 1)) {""")

    # 7. Init fields
    patch_file(aa_c,
        "  a->max_weight_norm = max_weight_norm;",
        """  a->max_weight_norm = max_weight_norm;
  a->trust_factor = trust_factor;
  a->r_adaptive = 1.0;""")

    # 8. Safeguard accept/reject feedback
    patch_file(aa_c,
        """    a->n_safeguard_reject++;
    aa_reset(a);
    TIME_TOC
    return -1;
  }
  TIME_TOC
  return 0;
}""",
        """    a->n_safeguard_reject++;
    trust_grow(a);
    aa_reset(a);
    TIME_TOC
    return -1;
  }
  trust_shrink(a);
  TIME_TOC
  return 0;
}""")

    # 9. aa.h header
    aa_h = SCS / "include" / "aa.h"
    patch_file(aa_h,
        """AaWork *aa_init(aa_int dim, aa_int mem, aa_int min_len, aa_int type1,
                aa_float regularization, aa_float relaxation,
                aa_float safeguard_factor, aa_float max_weight_norm,
                aa_int ir_max_steps, aa_int verbosity);""",
        """AaWork *aa_init(aa_int dim, aa_int mem, aa_int min_len, aa_int type1,
                aa_float regularization, aa_float relaxation,
                aa_float safeguard_factor, aa_float max_weight_norm,
                aa_float trust_factor, aa_int ir_max_steps,
                aa_int verbosity);""")

    # 10. scs.c — pass AA_TRUST_FACTOR
    scs_c = SCS / "src" / "scs.c"
    patch_file(scs_c,
        "                             AA_MAX_WEIGHT_NORM, AA_IR_MAX_STEPS,\n                             VERBOSITY))) {",
        "                             AA_MAX_WEIGHT_NORM, AA_TRUST_FACTOR,\n                             AA_IR_MAX_STEPS, VERBOSITY))) {")

    # 11. glbopts.h — define AA_TRUST_FACTOR
    patch_file(glb,
        """/* Reject AA steps whose weight vector exceeds this norm (prevents
 * numerically unstable extrapolation). */
#define AA_MAX_WEIGHT_NORM (1e10)""",
        """/* Reject AA steps whose weight vector exceeds this norm (prevents
 * numerically unstable extrapolation). */
#define AA_MAX_WEIGHT_NORM (1e10)
/* Opt-in trust region + adaptive r. 10.0 is tuned for ADMM/DRS; INFINITY
 * disables and recovers the original behavior. */
#define AA_TRUST_FACTOR (10.)""")

    print("scs-python patched with PR #54 changes.")


if __name__ == "__main__":
    main()
