#!/bin/bash
# run_synth.sh — v5/H2+H4 OOC 综合驱动（B8×R3C 集成）
# 用法: ./synth/run_synth.sh <pe|s16x4|s48|eng48|eng64> [stage]
#   stage = route（默认，含布线）/ place / synth（仅综合，大规模用）
# 工作区 E:\ae_syn\hb_h1（无空格路径），证据镜像回 results/<target>/
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WS="E:/ae_syn/hb_h1"
VIVADO="D:/software/Vivado/2021.2/bin/vivado.bat"
VAR="$1"; STAGE="${2:-route}"
[ -z "$VAR" ] && { echo "usage: $0 <pe|s16x4|s48|eng48|eng64> [stage]"; exit 1; }

case "$VAR" in
  pe)    FILES=(rtl/pack2_mult_dsp.sv rtl/ae_pe_p2.sv);                    TOP=ae_pe_p2;;
  s16x4) FILES=(rtl/pack2_mult_dsp.sv rtl/ae_pe_p2.sv
                rtl/ae_sysarr_p2.sv);                                      TOP=ae_sysarr_p2;;
  s48)   FILES=(rtl/pack2_mult_dsp.sv rtl/ae_pe_p2.sv
                rtl/ae_sysarr_p2.sv synth/wrap/top_s48.sv);                TOP=top_s48;;
  eng48) FILES=(rtl/pack2_mult_dsp.sv rtl/rq_ms_x.sv rtl/rq_v2.sv
                rtl/ae_pe_p2.sv rtl/ae_sysarr_p2.sv rtl/ae_gemm_p2.sv
                synth/wrap/top_eng48.sv);                                  TOP=top_eng48;;
  eng64) FILES=(rtl/pack2_mult_dsp.sv rtl/rq_ms_x.sv rtl/rq_v2.sv
                rtl/ae_pe_p2.sv rtl/ae_sysarr_p2.sv rtl/ae_gemm_p2.sv
                synth/wrap/top_eng64.sv);                                  TOP=top_eng64;;
  *) echo "unknown target $VAR"; exit 1;;
esac

SRC="$WS/src/$VAR"
mkdir -p "$SRC"
rm -f "$SRC"/*.sv
for f in "${FILES[@]}"; do cp "$ROOT/$f" "$SRC/"; done

OUT="$WS/runs/$VAR/$STAGE"
mkdir -p "$OUT"
echo "vivado syn.tcl $TOP 3.298 $OUT $SRC 8 $STAGE" > "$OUT/command.txt"
git -C "$ROOT" rev-parse HEAD > "$OUT/git_commit.txt" 2>/dev/null || echo "unknown" > "$OUT/git_commit.txt"

cd "$WS"
PERIOD=3.298
rc=1
for attempt in 1 2 3 4; do
  rm -rf "$WS/.Xil" 2>/dev/null || true
  cmd //c "$(cygpath -w "$VIVADO")" -mode batch -source "$ROOT/synth/tcl/syn.tcl" \
    -tclargs "$TOP" "$PERIOD" "$OUT" "$SRC" 8 "$STAGE" -notrace -nojournal -nolog \
    -log "$OUT/vivado.log" > "$OUT/synth_stdout.log" 2>&1 && rc=0 || rc=$?
  [ $rc -eq 0 ] && break
  echo "== synth attempt $attempt failed (transient?), retrying ==" >&2
done
[ $rc -ne 0 ] && { echo "== synth $VAR FAILED after retries =="; exit $rc; }

REPO_OUT="$ROOT/results/$VAR/$STAGE"
mkdir -p "$REPO_OUT"
cp "$OUT/command.txt" "$OUT/git_commit.txt" "$OUT/util.rpt" "$OUT/util_hier.rpt" \
   "$OUT/timing.rpt" "$OUT/timing_summary.rpt" "$OUT/wns.txt" "$OUT/power.rpt" \
   "$REPO_OUT/" 2>/dev/null || true
echo "== done $VAR stage=$STAGE =="
