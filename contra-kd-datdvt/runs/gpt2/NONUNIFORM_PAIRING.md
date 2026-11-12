# Fixed non-uniform layer-pairing control

Run:

```bash
bash project_commands.sh nonuniform
```

This experiment pairs student layers `[1, 3, 5, 6, 8, 10]` with teacher layers `[3, 7, 13, 21, 29, 38]`. The teacher indices are a fixed depth-warp schedule chosen before inspecting CKA or task scores. The official uniform teacher schedule is `[6, 13, 20, 27, 34, 41]`. The JSON pairing file is copied into each run for provenance.

The script pretrains a new velocity field and projector on GPU 6 with uniformly sampled `t`. Once pretraining finishes, DistiLLM trains and evaluates on GPU 6 while FDD trains and evaluates on GPU 7. The two jobs use separate torchrun and evaluation ports. Both student runs retain the original target curriculum and online velocity-field updates. Checkpoints, pairing, and per-seed evaluation results are under `results/gpt2/train/nonuniform_pairing_runs/<RUN_ID>/`; stdout logs are under `logs/gpt2_nonuniform_pairing/<RUN_ID>/`. Existing runs are not overwritten.

`project_commands.sh` now requires an explicit mode. This prevents an accidental no-argument call from launching an unrelated long experiment. The CKA experiment remains available as `bash project_commands.sh cka`.
