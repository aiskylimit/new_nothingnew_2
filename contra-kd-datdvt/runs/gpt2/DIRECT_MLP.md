# Direct teacher-state MLP ablation

Run on the server after preparing the same checkpoints and processed Dolly/OpenWebText data as `run_t0.sh`:

```bash
bash project_commands.sh
```

`project_commands.sh` activates the existing contra environment. It pretrains `results/gpt2/train/direct_state_mlp_t0/direct_state_mlp.pth` on physical GPU 6. Once that checkpoint exists, it trains and evaluates DistillM on GPU 6 and FDD on GPU 7 in parallel. Each child process sees one GPU as logical `cuda:0`. The two branches use separate torchrun and evaluation ports, logs, and output directories. The previously completed velocity `t=0` experiment is not run again.

The two models have **exactly the same architecture and trainable parameter count**: the same `VelocityField` backbone, dimensions, layer embedding, and fixed `t=0` input. The training target is the only change:

| Model | Regression target | Distillation target at curriculum fraction `c` |
| --- | --- | --- |
| Velocity at `t=0` | `h_T - P(h_S)` | `P(h_S) + c * v(P(h_S), 0, j)` |
| Direct MLP | `h_T` | `(1-c) * P(h_S) + c * f(P(h_S), 0, j)` |

At `c=1`, the direct MLP target is its prediction of the teacher state. Keeping the same curriculum fraction, training data, projector, optimizer, and evaluation protocol isolates the effect of direct state prediction. Because the velocity baseline is already trained only at `t=0`, this comparison tests the regression target and residual formulation, not the value of a full time-dependent flow.
