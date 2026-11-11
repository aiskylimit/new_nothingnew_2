# Layer-pairing ablation

For the current experiment, run `bash project_commands.sh` (or `bash project_commands.sh cka`) on the server. This is an end-to-end **CKA-selected pairing** run on physical GPU 6 only: CKA measurement, velocity-field pretraining, DistiLLM training and evaluation, then FDD training and evaluation. GPU 7 is unused. The velocity field is trained with `t` sampled uniformly and both students use the original target curriculum.

CKA is computed on one group of Dolly training records and used to select a strictly increasing teacher-layer schedule within eight layers of the official uniform schedule. A disjoint group of training records reports CKA for uniform and CKA-selected pairs. The selected schedule may equal the uniform schedule if that maximizes calibration CKA. The downstream evaluation sets are not used for pairing selection.

Outputs are isolated under `results/gpt2/train/cka_pairing_runs/<RUN_ID>/`: `cka/report.md`, `cka/cka_pairing.json`, `velocity_field/`, `distillm/`, `fdd/`, and `eval/{distillm,fdd}/`. Stdout logs are under `logs/gpt2_cka_pairing/<RUN_ID>/`. To run only the CKA diagnostic without training, use `bash project_commands.sh cka-only`.

The separate uniform-versus-fixed-non-uniform ablation script is `runs/gpt2/run_pairing_ablation.sh`; it is not called by `project_commands.sh` in this GPU-6-only phase.

## Later uniform versus fixed non-uniform comparison

The CKA diagnostic uses the teacher SFT and initialized student checkpoints on two disjoint groups of Dolly **training** records. It computes linear CKA on matching response-token positions for every teacher/student hidden-state pair. For this later comparison, CKA measures how well the official uniform pairs correspond; it does not select the fixed non-uniform alternative. Student layers stay at the official uniform indices. The fixed non-uniform teacher schedule applies a depth warp with exponent 1.5: `round(L_teacher * ((j+1)/(N+1))^1.5)`. For GPT-2 48→12 with six pairs, the uniform teacher layers are `[6, 13, 20, 27, 34, 41]` and the non-uniform layers are `[3, 7, 13, 21, 29, 38]`; both pair with student layers `[1, 3, 5, 6, 8, 10]`. The fixed alternative is specified without looking at CKA or downstream scores.

Two velocity fields are pretrained independently with identical settings except for the layer schedule. Uniform pairing runs on GPU 6 and fixed non-uniform pairing on GPU 7. After both fields finish, DistiLLM runs for both pairings in parallel on those GPUs, followed by FDD runs for both pairings in parallel. The two methods run in separate waves so each GPU hosts only one training job at a time. Each experiment is assigned a unique `RUN_ID`; existing output directories are rejected before writing.

Outputs:

- `results/gpt2/train/pairing_runs/<RUN_ID>/cka/`: full CKA matrices, the chosen pairing JSON, and a Markdown report.
- `results/gpt2/train/pairing_runs/<RUN_ID>/{uniform,nonuniform}/velocity_field/`: field and projector checkpoints.
- `results/gpt2/train/pairing_runs/<RUN_ID>/{uniform,nonuniform}/{distillm,fdd}/`: trained student checkpoints.
- `results/gpt2/train/pairing_runs/<RUN_ID>/{uniform,nonuniform}/eval/{distillm,fdd}/`: evaluation logs and predictions for four benchmarks and five generation seeds.
- `logs/gpt2_pairing_ablation/<RUN_ID>/`: CKA, pretraining, student training, and evaluation stdout logs.

CKA is descriptive evidence of representation similarity. The non-uniform pairing is fixed before measuring CKA, so its CKA and downstream ROUGE-L are independent comparisons. The five evaluation seeds vary generation only, not training initialization.
