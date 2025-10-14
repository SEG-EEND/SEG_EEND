#!/usr/bin/env python3

import torch.optim as optim
from torch.nn import Module
from types import SimpleNamespace
from typing import Any, Dict
import torch.distributed as dist
import logging
import math

def _ddp_world_size() -> int:
    """Returns the world size in a DDP environment, or 1 if not applicable."""
    return dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1

class NoamOpt:
    """
    Noam LR schedule wrapper (https://arxiv.org/abs/1706.03762).
    Implements learning rate scaling for multi-GPU training.

    lrate = model_size**(-0.5) * min(step**(-0.5), step * warmup**(-1.5))
    """
    def __init__(
        self,
        model_size: int,
        warmup: int,
        optimizer: optim.Optimizer,
        step_scale: int = 1,
        lr_scale: float = 1.0,
    ) -> None:
        self.optimizer = optimizer
        self._step = 0
        self.warmup = int(max(1, warmup))
        self.model_size = int(model_size)
        self._rate = 0.0
        self.step_scale = max(1, int(step_scale))
        self.lr_scale = float(lr_scale)

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the scheduler."""
        return {k: v for k, v in self.__dict__.items() if k != "optimizer"}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the scheduler's state."""
        self.__dict__.update(state_dict)

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zeros the gradients of the optimizer."""
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        """Update parameters and learning rate."""
        self._step += 1
        lr = self.rate()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self._rate = lr
        self.optimizer.step()

    def rate(self, step: int | None = None) -> float:
        """Calculates the learning rate for a given step."""
        if step is None:
            step = self._step
        
        # Effective step considers scaling for multi-GPU timing alignment
        eff_step = max(1, int(step) * self.step_scale)
        
        base_lr = (self.model_size ** -0.5) * min(
            eff_step ** -0.5, eff_step * (self.warmup ** -1.5)
        )
        
        # Apply final LR scaling (e.g., linear or sqrt)
        return base_lr * self.lr_scale

    def get_rate(self) -> float:
        """Returns the current learning rate."""
        return float(self._rate)


def setup_optimizer(args: SimpleNamespace, model: Module) -> optim.Optimizer:
    """
    Sets up the optimizer based on provided arguments.

    args:
      - optimizer (str): "adam", "sgd", or "noam".
      - lr (float): Learning rate for adam/sgd.
      - hidden_size (int): Model size for noam scheduler.
      - noam_warmup_steps (int): Base warmup steps for noam.
      - accum_steps (int): Gradient accumulation steps.
      - noam_scale_rule (str): Scaling rule for multi-GPU.
          - "linear", "sqrt", "step", "hybrid"
      - noam_scale_warmup (bool): If True, scales down warmup steps.
    """
    opt_name = getattr(args, "optimizer", "adam").lower()

    if opt_name == "adam":
        return optim.Adam(model.parameters(), lr=getattr(args, "lr", 1e-3))

    if opt_name == "sgd":
        return optim.SGD(model.parameters(), lr=getattr(args, "lr", 1e-2))

    if opt_name == "noam":
        model_size = int(getattr(args, "hidden_size", 256))
        warmup_steps = int(getattr(args, "noam_warmup_steps", 100000))
        
        world_size = _ddp_world_size()
        accum_steps = max(1, int(getattr(args, "accum_steps", 1)))
        k = world_size * accum_steps

        # ---- calculate k ----
        k_override = int(getattr(args, "noam_k", 0))
        k = k_override if k_override > 0 else (world_size * accum_steps)
        
        base_opt = optim.Adam(
            model.parameters(),
            lr=0.0,  # LR is controlled by NoamOpt wrapper
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        # --- Multi-GPU Scaling Logic ---
        scale_rule = getattr(args, "noam_scale_rule", "sqrt").lower()
        scale_warmup = bool(getattr(args, "noam_scale_warmup", False))

        # ---- noam parameters ----
        alpha = getattr(args, "noam_alpha", None)              # float | None
        lr_scale_fixed = getattr(args, "noam_lr_scale", None)  # float | None
        step_scale_override = int(getattr(args, "noam_step_scale", 0))
        warmup_scale = float(getattr(args, "noam_warmup_scale", 1.0))
        
        lr_scale = 1.0
        step_scale = 1
        
        if k > 1: # Only apply scaling in multi-GPU/accumulation settings
            if scale_rule == "linear":
                lr_scale = float(k)
                logging.info(f"🚀 Using Noam with [Linear Scaling] rule. LR scale: {lr_scale:.2f}")
            elif scale_rule == "sqrt":
                lr_scale = math.sqrt(k)
                logging.info(f"💡 Using Noam with [Square Root Scaling] rule. LR scale: {lr_scale:.2f}")
            elif scale_rule == "step":
                step_scale = k
                logging.warning(f"⚠️ Using Noam with [Step Scaling] rule. This is generally not recommended.")
            elif scale_rule == "hybrid":
                # This is an experimental, non-standard approach.
                # Combines fast schedule progression with LR value scaling.
                step_scale = k
                lr_scale = math.sqrt(k)
                logging.info("🧪 Using Noam with [Hybrid Scaling] rule. step_scale=%d, lr_scale=%.2f", step_scale, lr_scale)
            else:
                raise ValueError(f"Unknown noam_scale_rule: {scale_rule}")
        else:
            logging.info("Running in single-GPU mode (k=1). No LR scaling applied.")


        final_warmup = warmup_steps
        if scale_warmup and k > 1:
            final_warmup = max(1, int(round(warmup_steps / k)))
            logging.info(f"Warmup scaling ON: {warmup_steps} -> {final_warmup} steps.")
        else:
            logging.info(f"Warmup scaling OFF: Using original {final_warmup} steps for stability.")

        return NoamOpt(
            model_size=model_size,
            warmup=final_warmup,
            optimizer=base_opt,
            step_scale=step_scale,
            lr_scale=lr_scale,
        )

    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def get_rate(optimizer: optim.Optimizer) -> float:
    """Retrieves the current learning rate from the optimizer."""
    if isinstance(optimizer, NoamOpt):
        return optimizer.get_rate()
    # Fallback for other optimizers
    for pg in optimizer.param_groups:
        return float(pg.get("lr", 0.0))
    return 0.0