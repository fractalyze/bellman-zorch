"""Export the fused bellman Groth16 GPU core: h-FFT + the 5 MSMs in one
executable, using bellman's exact h pipeline (generator=7, gnark-style
coefficient tail, no bit-reverse).

The core takes the prover witness + the (dense, unfiltered) proving-key point
arrays and returns the 5 group elements bellman's final assembly needs:

    core(z_std, az_std, bz_std, A_q, Bg1_q, Bg2_q, L_q, H_q)
      -> (msm_A:G1, msm_Bg1:G1, msm_Bg2:G2, msm_L:G1, msm_h:G1)

where z_std = [inputs ‖ aux] (full assignment), az/bz = A·z / B·z evaluation
vectors (padded to n), and the queries are bellman's CRS points in dense order.

`lax.ntt`/`lax.msm` lower shape-specialized, so an executable is fixed to one
(n, m, num_inputs) shape; the point VALUES are runtime inputs. Run with the
matched frx 0.10 venv (see the README for the install):

    FRX_PLATFORMS=cuda,cpu .venv/bin/python \
        export/export_bellman_core.py <n> <m> <num_inputs>
"""
import io
import os
import sys
from pathlib import Path

import frx
import frx.numpy as fnp
import numpy as np
from frx import lax
from frx.lax import NttType  # frx exposes the field transform as lax.ntt
from zk_dtypes import bn254_g1_affine, bn254_g2_affine, bn254_sf, bn254_sf_mont

G = bn254_sf_mont(7)  # halo2curves BN254 Fr MULTIPLICATIVE_GENERATOR


def _transform(x, n, inverse):
    """Forward/inverse field transform: `lax.ntt` NTT/INTT with the generator-7
    root in natural order (no bit-reverse) — bellman's exact h-FFT convention."""
    kind = NttType.INTT if inverse else NttType.NTT
    return lax.ntt(x, ntt_type=kind, ntt_length=n, generator=G)

ART = Path(
    os.environ.get(
        "BELLMAN_ZORCH_ARTIFACTS",
        str(Path(__file__).resolve().parent.parent / "artifacts"),
    )
)


def make_core(n: int, m: int, num_inputs: int):
    # The dtype carries the field, so powers and inverses are plain operators.
    shift = fnp.array([G**i for i in range(n)], dtype=bn254_sf_mont)
    inv_shift = fnp.array([G**-i for i in range(n)], dtype=bn254_sf_mont)
    den = fnp.array((G**n - 1) ** -1, dtype=bn254_sf_mont)  # 1/Z on the coset

    def h_fft(az_std, bz_std):
        az = lax.convert_element_type(az_std, bn254_sf_mont)
        bz = lax.convert_element_type(bz_std, bn254_sf_mont)
        cz = az * bz
        a = _transform(az, n, inverse=True)
        b = _transform(bz, n, inverse=True)
        c = _transform(cz, n, inverse=True)
        ac = _transform(a * shift, n, inverse=False)
        bc = _transform(b * shift, n, inverse=False)
        cc = _transform(c * shift, n, inverse=False)
        h_evals = (ac * bc - cc) * den
        h_poly = _transform(h_evals, n, inverse=True)
        return lax.convert_element_type(h_poly * inv_shift, bn254_sf)

    @frx.jit
    def core(z_std, az_std, bz_std, a_q, bg1_q, bg2_q, l_q, h_q):
        h = h_fft(az_std, bz_std)[: n - 1]
        aux = z_std[num_inputs:]
        return (
            lax.msm(z_std, a_q),
            lax.msm(z_std, bg1_q),
            lax.msm(z_std, bg2_q),
            lax.msm(aux, l_q),
            lax.msm(h, h_q),
        )

    return core


def write_bytecode(lowered, path):
    m = lowered.compiler_ir(dialect="stablehlo")
    try:
        from frx._src.interpreters import mlir as _fmlir

        data = _fmlir.module_to_bytecode(m)
    except Exception:
        buf = io.BytesIO()
        m.operation.write_bytecode(buf)
        data = buf.getvalue()
    Path(path).write_bytes(data)


def main():
    n, m, num_inputs = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    num_aux = m - num_inputs
    core = make_core(n, m, num_inputs)
    args = (
        np.zeros(m, dtype=bn254_sf),  # z_std
        np.zeros(n, dtype=bn254_sf),  # az
        np.zeros(n, dtype=bn254_sf),  # bz
        np.zeros(m, dtype=bn254_g1_affine),  # A_q
        np.zeros(m, dtype=bn254_g1_affine),  # Bg1_q
        np.zeros(m, dtype=bn254_g2_affine),  # Bg2_q
        np.zeros(num_aux, dtype=bn254_g1_affine),  # L_q
        np.zeros(n - 1, dtype=bn254_g1_affine),  # H_q
    )
    lowered = core.lower(*args)
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / f"bellman_core_n{n}_m{m}_i{num_inputs}.mlirbc"
    write_bytecode(lowered, out)
    print(f"wrote {out} ({out.stat().st_size} B); n={n} m={m} num_inputs={num_inputs}")


if __name__ == "__main__":
    main()
