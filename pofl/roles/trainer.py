from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pofl.fl.pool_formation import TrainerProfile


@dataclass
class TrainerNode:
    trainer_id: str
    profile: TrainerProfile
    X: torch.Tensor
    y: torch.Tensor

    @staticmethod
    def synthetic_profile(
        trainer_id: str,
        rng: np.random.Generator,
        num_samples: int,
        num_classes: int,
        input_dim: int,
        delay: float,
        label_bias: int,
    ) -> TrainerNode:
        """Non-IID: shift label distribution toward label_bias."""
        logits = rng.standard_normal((num_classes,))
        logits[label_bias] += 2.0
        p = np.exp(logits - logits.max())
        p = p / p.sum()
        labels = rng.choice(num_classes, size=num_samples, p=p)
        X = rng.standard_normal((num_samples, input_dim)).astype(np.float32)
        pr = TrainerProfile(
            trainer_id=trainer_id,
            sample_count=num_samples,
            delay=delay,
            labels=labels,
            num_classes=num_classes,
        )
        return TrainerNode(
            trainer_id=trainer_id,
            profile=pr,
            X=torch.tensor(X),
            y=torch.tensor(labels, dtype=torch.long),
        )
