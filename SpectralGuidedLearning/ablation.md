[2026-10-08 15:32:02] === summary ===
# Answer-gain ablation summary


Track `r1-qwen-1.5b-palign`, results from `results`. `-` means not run / not finished.


## Main ablation (training seed 42)


| Method | AIME24 p@1 | AIME25 p@1 | AMC12 p@1 | MATH500 p@1 | Avg p@1 | Avg p@3 |
|---|---:|---:|---:|---:|---:|---:|
| E0 Uniform SFT | 21.11% | 15.56% | 51.00% | 77.73% | 41.35% | 51.70% |
| E1 ALG | 18.89% | 14.44% | 50.60% | 78.13% | 40.52% | 51.75% |
| E2 Predictability-Easy | 18.89% | 20.00% | 55.02% | 77.27% | 42.79% | 53.67% |
| E3 Predictability-Hard | 20.00% | 20.00% | 50.60% | 76.93% | 41.88% | 53.42% |
| E4 Random Assignment | 17.78% | 13.33% | 53.41% | 77.87% | 40.60% | 49.48% |
| E5 ALG-NoAnswerUpweight | 14.44% | 17.78% | 52.21% | 77.60% | 40.51% | 53.74% |


## Hyperparameter sensitivity (seed 42)


| Parameter | Value | Arm | Avg p@1 | Avg p@3 | w P10 | w median | w P90 | steps w>1 | top-10% mass |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| lambda | 0 | E0 Uniform SFT | 41.35% | 51.70% | - | - | - | - | - |
| lambda | 0.5 | E1 ALG | 40.52% | 51.75% | 0.746 | 0.908 | 1.180 | 27.20% | 29.65% |
| lambda | 1 | E6 ALG lambda=1 | 41.15% | 51.20% | 0.491 | 0.816 | 1.361 | 27.20% | 38.59% |
| tau | 1 | E7 ALG tau=1 | 42.78% | 54.70% | 0.595 | 0.761 | 1.233 | 18.42% | 39.45% |
| tau | 2 | E1 ALG | 40.52% | 51.75% | 0.746 | 0.908 | 1.180 | 27.20% | 29.65% |
| tau | 4 | E8 ALG tau=4 | 40.34% | 47.28% | 0.863 | 0.966 | 1.102 | 33.13% | 24.88% |


## Training-seed robustness (mean ± std over finished seeds)


| Method | Avg p@1 | Avg p@3 | Seeds finished |
|---|---:|---:|---|
| (needs at least two finished seeds) | - | - | - |


## Missing seed-42 runs


none


## Run status (this launcher invocation)


| Arm | Stages | GPU | Status | Seconds | Log |
|---|---|---:|---|---:|---|
| e0-uniform-r1-qwen-1.5b-palign | prepare,vanilla | 0 | ok | 11 | logs/e0-uniform-r1-qwen-1.5b-palign-*.log |
| e1-alg-r1-qwen-1.5b-palign | answer_gain | 0 | ok | 0 | logs/e1-alg-r1-qwen-1.5b-palign-*.log |
| e2-predictability-easy-r1-qwen-1.5b-palign | predictability | 1 | ok | 9 | logs/e2-predictability-easy-r1-qwen-1.5b-palign-*.log |
| e1-alg-r1-qwen-1.5b-palign | gain_signal,weights | 0 | ok | 1 | logs/e1-alg-r1-qwen-1.5b-palign-*.log |
| e2-predictability-easy-r1-qwen-1.5b-palign | score_signal,weights | 0 | ok | 2 | logs/e2-predictability-easy-r1-qwen-1.5b-palign-*.log |
| e3-predictability-hard-r1-qwen-1.5b-palign | score_transform,score_signal,weights | 0 | ok | 1 | logs/e3-predictability-hard-r1-qwen-1.5b-palign-*.log |
| e4-random-assignment-r1-qwen-1.5b-palign | score_transform,gain_signal,weights | 0 | ok | 1 | logs/e4-random-assignment-r1-qwen-1.5b-palign-*.log |
| e5-alg-no-answer-upweight-r1-qwen-1.5b-palign | answer_only,weights | 0 | ok | 10 | logs/e5-alg-no-answer-upweight-r1-qwen-1.5b-palign-*.log |
| e6-alg-lambda1-r1-qwen-1.5b-palign | weights | 0 | ok | 1 | logs/e6-alg-lambda1-r1-qwen-1.5b-palign-*.log |
| e7-alg-tau1-r1-qwen-1.5b-palign | weights | 0 | ok | 1 | logs/e7-alg-tau1-r1-qwen-1.5b-palign-*.log |
| e8-alg-tau4-r1-qwen-1.5b-palign | weights | 0 | ok | 2 | logs/e8-alg-tau4-r1-qwen-1.5b-palign-*.log |
| e4-random-assignment-r1-qwen-1.5b-palign | score_analysis | 0 | ok | 0 | logs/e4-random-assignment-r1-qwen-1.5b-palign-*.log |
| e0-uniform-r1-qwen-1.5b-palign | train,eval | 0 | ok | 2985 | logs/e0-uniform-r1-qwen-1.5b-palign-*.log |
| e3-predictability-hard-r1-qwen-1.5b-palign | train,eval | 3 | ok | 3031 | logs/e3-predictability-hard-r1-qwen-1.5b-palign-*.log |
| e2-predictability-easy-r1-qwen-1.5b-palign | train,eval | 2 | ok | 3035 | logs/e2-predictability-easy-r1-qwen-1.5b-palign-*.log |
| e1-alg-r1-qwen-1.5b-palign | train,eval | 1 | ok | 3774 | logs/e1-alg-r1-qwen-1.5b-palign-*.log |
| e4-random-assignment-r1-qwen-1.5b-palign | train,eval | 0 | ok | 2965 | logs/e4-random-assignment-r1-qwen-1.5b-palign-*.log |
| e6-alg-lambda1-r1-qwen-1.5b-palign | train,eval | 2 | ok | 2997 | logs/e6-alg-lambda1-r1-qwen-1.5b-palign-*.log |
| e7-alg-tau1-r1-qwen-1.5b-palign | train,eval | 3 | ok | 3006 | logs/e7-alg-tau1-r1-qwen-1.5b-palign-*.log |
| e5-alg-no-answer-upweight-r1-qwen-1.5b-palign | train,eval | 1 | ok | 4043 | logs/e5-alg-no-answer-upweight-r1-qwen-1.5b-palign-*.log |
| e8-alg-tau4-r1-qwen-1.5b-palign | train,eval | 0 | ok | 2847 | logs/e8-alg-tau4-r1-qwen-1.5b-palign-*.log |


summary -> experiments/answer_gain/results/ablation_summary.md (+ per_benchmark.csv, seed_summary.csv)
[2026-10-08 15:32:03] === finished mode=all: exit 0, 0 failed run(s); master log logs/answer-gain-all-20261008-130446.log ===
