#!/usr/bin/env bash
# Reproduce the GPU-core-vs-bellman-CPU benchmark (examples/bench.rs): export a
# fused core for each MiMC size, then run the sweep. Pass round counts as args,
# else the default 2^13..2^18 sweep is used.
#
# Requires the matched frx 0.10 export stack (see README) — e.g.:
#   XLA_PJRT_PLUGIN=$PWD/.venv/lib/python3.11/site-packages/frx_plugins/xla_cuda12/xla_cuda_plugin.so \
#   XLA_VENV_PYTHON=$PWD/.venv/bin/python ./bench.sh
set -euo pipefail
cd "$(dirname "$0")"
: "${XLA_PJRT_PLUGIN:?set XLA_PJRT_PLUGIN to the frx-cuda12 xla_cuda_plugin.so}"
: "${XLA_VENV_PYTHON:?set XLA_VENV_PYTHON to the matched frx 0.10 export venv python}"

ROUNDS="${*:-4000 8000 16000 32000 64000 130000}"

# Export one fused core per size: n = next_pow2(2*rounds+2), m = 2*rounds+3.
for r in $ROUNDS; do
  nc=$((2 * r + 2)); n=1; while [ "$n" -lt "$nc" ]; do n=$((n * 2)); done
  m=$((2 * r + 3))
  if [ ! -f "artifacts/bellman_core_n${n}_m${m}_i2.mlirbc" ]; then
    echo ">>> exporting core n=$n m=$m (rounds=$r)" >&2
    FRX_PLATFORMS=cuda,cpu "$XLA_VENV_PYTHON" export/export_bellman_core.py "$n" "$m" 2 >/dev/null
  fi
done

exec cargo run --release --example bench -- $ROUNDS
