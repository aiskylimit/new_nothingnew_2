# Velocity loss-weight ramp ablation

Run `bash project_commands.sh` on the server. It pretrains a separate velocity field with `t` sampled uniformly from `[0,1]`, matching the `submission-neurips` branch. The completed `t=0` checkpoint is not reused or overwritten.

The original `submission-neurips` curriculum uses `c=(epoch+1)/epochs` in the target `P(h_S)+c*v(P(h_S),0,j)` and gives the Contra loss full weight. This ablation uses the full target `P(h_S)+v(P(h_S),0,j)` in every epoch and multiplies only the Contra loss by `c`. The other distillation losses and `kd_ratio` stay as before. The default `--representation-model velocity` and default `--contra-curriculum target` preserve the original behavior for other runs.

Pretraining runs on GPU 6. Once its checkpoint is ready, DistillM runs on GPU 6 and FDD runs on GPU 7 in parallel. Each invocation gets a unique `RUN_ID` (timestamp and process ID). Logs go to `logs/gpt2_loss_ramp_ablation/<RUN_ID>/{pretrain,distillm,fdd}.log`; checkpoints and evaluation results go to `results/gpt2/train/loss_ramp_runs/<RUN_ID>/`. An existing `RUN_ID` is rejected before any output is written.
