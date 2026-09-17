"""Phase 6: Algorithm 1, rounded down to TROPIC-P (alpha=0, mu=0, fixed epsilon).

    for outer_iter in range(K):
        o ~ student.generate(x)                      # line 3, "student acts"
        theta_k <- theta ; cache pi^old               # line 4
        cache pi^T (== pi-bar^T since alpha=0)         # line 5, "teacher grades"
        for t: bisect s_t, cache pi^eps_t              # lines 6-13
        for inner_epoch in range(E):                   # lines 15-18
            pi^S <- forward_student(...)  (WITH grad)
            loss <- D^(beta)(pi^S || pi^eps)
            backward / optimizer step

Batch size is fixed to 1 example per outer iteration: teacher and student
prefixes have different lengths, and padding two ragged left-contexts inside
one batch on top of a from-scratch training loop is exactly the kind of bug
this smoke test is meant to avoid (see plan.md, Phase 3 risk note). Batching
is future work once this pipeline is verified correct end to end.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tropic.data import load_gsm8k_examples
from tropic.loss import privileged_gap, tropic_p_loss
from tropic.model import ContextualPolicy
from tropic.primitives import kl_categorical
from tropic.rollout import generate_rollout


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TROPIC-P smoke run on Qwen2.5-0.5B-Instruct")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--num_examples", type=int, default=8)
    p.add_argument("--outer_iters", type=int, default=3)
    p.add_argument("--inner_epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--delta", type=float, default=1e-5, help="kept for parity with the plan; top_k already caps mass")
    p.add_argument("--default_mass", type=float, default=1e-5)
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_file", default="tropic_smoke_run.jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float32)
    model.train()

    examples = load_gsm8k_examples(tokenizer, num_examples=args.num_examples, seed=args.seed)
    print(f"Loaded {len(examples)} GSM8K examples.")

    policy = ContextualPolicy(model, tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    log_path = Path(args.log_file)
    log_records = []

    for outer_iter in range(args.outer_iters):
        t0 = time.time()
        ex = examples[outer_iter % len(examples)]

        # line 3: student acts (on-policy, no grad)
        generated_ids, gen_len = generate_rollout(
            model,
            tokenizer,
            ex.student_prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        # line 4: theta_k <- theta, cache pi^old (frozen for the whole outer iteration)
        log_old = policy.forward_checkpoint(ex.student_prompt_ids, generated_ids)

        # line 5: teacher grades (alpha=0 -> pi-bar^T == pi^T, no extra forward pass)
        log_teacher = policy.forward_teacher(ex.teacher_prefix_ids, generated_ids)

        inner_losses = []
        s_last = None
        for inner_epoch in range(args.inner_epochs):
            log_student = policy.forward_student(ex.student_prompt_ids, generated_ids)

            loss, s = tropic_p_loss(
                log_student,
                log_teacher,
                log_old,
                generated_ids,
                epsilon=args.epsilon,
                beta=args.beta,
                top_k=args.top_k,
                default_mass=args.default_mass,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            inner_losses.append(loss.item())
            s_last = s

        with torch.no_grad():
            log_student_final = policy.forward_student(ex.student_prompt_ids, generated_ids)
            drift = kl_categorical(log_student_final, log_old).mean().item()
            gap = privileged_gap(log_old, log_teacher).item()

        record = {
            "outer_iter": outer_iter,
            "question": ex.question[:80],
            "gen_len": gen_len,
            "loss": inner_losses[-1],
            "loss_first_inner_epoch": inner_losses[0],
            "mean_s": s_last.mean().item(),
            "frac_s_equal_1": (s_last >= 1.0 - 1e-6).float().mean().item(),
            "kl_student_old_drift": drift,
            "privileged_gap": gap,
            "elapsed_sec": time.time() - t0,
        }
        log_records.append(record)
        print(json.dumps(record))

        if not (record["loss"] == record["loss"]):  # NaN check
            raise RuntimeError(f"loss became NaN at outer_iter={outer_iter}")

    with log_path.open("w") as f:
        for r in log_records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(log_records)} records to {log_path}")


if __name__ == "__main__":
    main()
