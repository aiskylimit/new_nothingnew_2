"""Disk-backed last-batch targets, bounded by one example in host/GPU RAM."""

from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F


class ReplayCache:
    def __init__(self, directory=""):
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="graph-replay-", dir=directory or None)
        self.path = Path(self.temporary.name)
        self.rows = []
        self.bytes = 0

    def add(self, example, tokens, targets):
        files = []
        for index, target in enumerate(targets):
            path = self.path / f"{len(self.rows)}-{index}.pt"
            torch.save(target.detach().float().cpu(), path)
            self.bytes += path.stat().st_size
            files.append(path)
        self.rows.append((example, list(tokens), files))

    def loss(self, trainer):
        trainer.model.eval()
        total = torch.zeros((), device=trainer.device, dtype=torch.float32)
        with trainer.timed("last_student_replay"), torch.no_grad():
            for example, tokens, files in self.rows:
                logits = trainer.logits(example, tokens)
                for chunk, path in zip(trainer.chunks(len(tokens)), files):
                    q = torch.load(path, map_location=trainer.device, weights_only=True)
                    logp = F.log_softmax(logits[chunk].float() / trainer.config.distillation_temperature, -1)
                    total += (
                        (torch.xlogy(q, q) - q * logp).sum()
                        / len(tokens)
                        / trainer.config.effective_batch_size
                    )
                    del q, logp
                trainer.count("replay_student_forward_calls")
                trainer.count("replay_student_scored_tokens", len(trainer.prompt(example)) + len(tokens))
                del logits
        if trainer.world > 1:
            torch.distributed.all_reduce(total)
        return float(total)

    def close(self):
        self.temporary.cleanup()
