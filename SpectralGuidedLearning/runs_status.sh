#!/usr/bin/env bash
# Which runs of phase 1 and phase 2 are done? Reads only files on disk (and ps), changes nothing; safe while
# runs are going.
#   bash runs_status.sh
# Phase 1 runs 2-9 were killed and dropped ("dropped" in the Plan column): whatever they finished is still shown.
# State per run: done (eval summary written; W4/hard/R6 also finalized), eval (final checkpoint, no scores yet),
# train (epoch k/3 checkpoints so far), - (not started). Avg = mean over the benchmarks of pass@1 / pass@3.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}/bin/python"
[[ -x "${PY}" ]] || PY=python3

W4=results/w4/runs HARD=results/hard/runs
R=rebuttal/results
# run | phase | plan | GPU | description | eval dir (summary.json) | final checkpoint | epoch checkpoints dir | finalize marker
# (zero-shot runs evaluate the base model: no checkpoint)
RUNS=(
  "1|1+2|run|0|1.5B ALERD, s1K-1.1|rebuttal/results/iwc-gain-l05-r1-qwen-1.5b-s1k|checkpoints/iwc-gain-l05-r1-qwen-1.5b-s1k||"
  "2|1|dropped|0|1.5B SFT, s1K-1.1|${R}/sft-nll-r1-qwen-1.5b-s1k|checkpoints/sft-nll-r1-qwen-1.5b-s1k||"
  "3|1|dropped|0|1.5B zero-shot, s1K-1.1|${R}/zeroshot-r1-qwen-1.5b-s1k|||"
  "4|1|dropped|1|7B ALERD LoRA, s1K-1.1|${R}/iwc-gain-l05-lora-qwen25-7b-s1k|checkpoints/iwc-gain-l05-lora-qwen25-7b-s1k||"
  "5|1|dropped|1|7B SFT LoRA, s1K-1.1|${R}/sft-nll-lora-qwen25-7b-s1k|checkpoints/sft-nll-lora-qwen25-7b-s1k||"
  "6|1|dropped|1|7B zero-shot, s1K-1.1|${R}/zeroshot-qwen25-7b-s1k|||"
  "7|1|dropped|6|Qwen3-8B ALERD LoRA, s1K|${R}/iwc-gain-l05-lora-qwen3-8b-s1k|checkpoints/iwc-gain-l05-lora-qwen3-8b-s1k||"
  "8|1|dropped|6|Qwen3-8B SFT LoRA, s1K|${R}/sft-nll-lora-qwen3-8b-s1k|checkpoints/sft-nll-lora-qwen3-8b-s1k||"
  "9|1|dropped|6|Qwen3-8B zero-shot, s1K|${R}/zeroshot-qwen3-8b-s1k|||"
  "10|2|run|1|1.5B answer-steps, s1K-1.1|rebuttal/results/ansstep-mention-r1-qwen-1.5b-s1k|checkpoints/ansstep-mention-r1-qwen-1.5b-s1k||"
  "13|1+2|run|6|ASFT, 966 traces|rebuttal/results/sft-asft-r1-qwen-1.5b-palign|checkpoints/sft-asft-r1-qwen-1.5b-palign||"
  "14|1|run|7|VCORE, 966 traces|rebuttal/results/sft-vcore-r1-qwen-1.5b-palign|checkpoints/sft-vcore-r1-qwen-1.5b-palign||"
  "15|2|run|6|positional R6|results/positional/runs/r6|results/positional/runs/r6/checkpoint||results/positional/runs/r6/eval/metrics.json"
  "18|2|run|7|W4 ALERD P1/S0|${W4}/r1q1.5b__alg__P1__S0__tau1_lam0.5_c2__seed42|${W4}/r1q1.5b__alg__P1__S0__tau1_lam0.5_c2__seed42/checkpoints/epoch3|${W4}/r1q1.5b__alg__P1__S0__tau1_lam0.5_c2__seed42/checkpoints|${W4}/r1q1.5b__alg__P1__S0__tau1_lam0.5_c2__seed42/run_check.json"
  "20|2|run|7|W4 ALERD P0/S1|${W4}/r1q1.5b__alg__P0__S1__tau1_lam0.5_c2__seed42|${W4}/r1q1.5b__alg__P0__S1__tau1_lam0.5_c2__seed42/checkpoints/epoch3|${W4}/r1q1.5b__alg__P0__S1__tau1_lam0.5_c2__seed42/checkpoints|${W4}/r1q1.5b__alg__P0__S1__tau1_lam0.5_c2__seed42/run_check.json"
  "30|2|run|1|ALG-hard h59|${HARD}/r1q1.5b__h59__P0__S0__seed42|${HARD}/r1q1.5b__h59__P0__S0__seed42/checkpoints/epoch3|${HARD}/r1q1.5b__h59__P0__S0__seed42/checkpoints|${HARD}/r1q1.5b__h59__P0__S0__seed42/run_check.json"
)

scores() {  # <summary.json>: "Avg p@1 / p@3" over its benchmarks
  "${PY}" - "$1" <<'EOF'
import json, sys
rows = json.load(open(sys.argv[1]))
p1 = sum(r["pass@1"] for r in rows) / len(rows)
k3 = [r["pass@3"] for r in rows if "pass@3" in r]
print(f"{100 * p1:.2f} / {100 * sum(k3) / len(k3):.2f}" if k3 else f"{100 * p1:.2f}", f"({len(rows)} bench)")
EOF
}

printf '%-4s %-6s %-8s %-4s %-28s %-12s %s\n' Run Phase Plan GPU "Description" State "Avg p@1 / p@3"
printf '%s\n' "--------------------------------------------------------------------------------------------------"
for entry in "${RUNS[@]}"; do
  IFS='|' read -r id phase plan gpu desc evaldir final epochs marker <<< "${entry}"
  state="-" score=""
  if [[ -f "${evaldir}/summary.json" ]]; then
    state=done score="$(scores "${evaldir}/summary.json" 2>/dev/null || echo '?')"
    [[ -z "${marker}" || -f "${marker}" ]] || state="done*"   # scored, report step still pending
  elif [[ -n "${final}" && ( -f "${final}/config.json" || -f "${final}/adapter_config.json" ) ]]; then
    state=eval
  else
    n=0
    if [[ -n "${epochs}" ]]; then
      for e in 1 2 3; do [[ -f "${epochs}/epoch${e}/config.json" ]] && n=${e}; done
    else
      n=$(find "${final}" -maxdepth 1 -name 'checkpoint-*' -type d 2>/dev/null | wc -l)
    fi
    [[ ( -n "${final}" && -d "${final}" ) || ( -n "${epochs}" && -d "${epochs}" ) ]] && state="train ${n}/3"
    [[ -z "${final}" && -d "${evaldir}/raw" ]] && state=eval   # zero-shot: generations started
  fi
  printf '%-4s %-6s %-8s %-4s %-28s %-12s %s\n' "${id}" "${phase}" "${plan}" "${gpu}" "${desc}" "${state}" "${score}"
done
echo "Phase 1+2: started in phase 1, finished by phase 2. A killed run keeps 'train k/3' or 'eval' forever."
echo "done* = scored, but the run's report step (run_check.json / eval/metrics.json) has not finished"

echo
echo "== analyses"
for f in results/positional/analysis/a2_summary.json results/positional/decision.json \
         results/w4/analysis/table_r_w4a.md results/w4/analysis/table_r_w4b.md \
         results/hard/selection/alg-h59/manifest.json results/hard/analysis/table_r_h1.md \
         rebuttal/results/comparison-table.md; do
  [[ -f "${f}" ]] && printf '  ok  %s  (%s)\n' "${f}" "$(date -r "${f}" '+%F %T')" || printf '  --  %s\n' "${f}"
done

echo
echo "== running now"
ps -eo pid=,etime=,args= | awk '
  /sgl\.(training\.train|eval\.evaluate|signals\.answer_gain|robustness\.token_logp|hardsel\.nll_eval)/ && !/awk/ {
    what = ""; gpu = ""
    for (i = 3; i <= NF; i++) {
      if ($i ~ /^-m$/ && what == "") what = $(i + 1)
      if ($i ~ /^--(output-dir|tag|output)$/) what = what " " $(i + 1)
      if ($i ~ /^CUDA_VISIBLE_DEVICES=/) gpu = $i
    }
    printf "  pid %-8s up %-12s %s\n", $1, $2, what; n++
  }
  END { if (!n) print "  (nothing)" }'
echo
for out in phase1.out phase2.out; do
  echo
  echo "== last lines of ${out}"
  [[ -f "${out}" ]] && tail -n 8 "${out}" || echo "  (no ${out} here)"
done
