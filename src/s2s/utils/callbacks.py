import pytorch_lightning as pl
import torch

class GradientMonitor(pl.callbacks.Callback):
    def __init__(self, log_every_n_steps=100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_after_backward(self, trainer, pl_module):
        if trainer.global_step % self.log_every_n_steps == 0:
            gradient_norm = 0.0
            max_norm = 0.0
            for p in pl_module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    gradient_norm += param_norm.item() ** 2
                    max_norm = max(max_norm, param_norm.item())
            
            gradient_norm = gradient_norm ** 0.5
            
            # Log gradient norms
            trainer.logger.log_metrics(
                {
                    "gradient/norm": gradient_norm,
                    "gradient/max": max_norm
                },
                step=trainer.global_step
            )
            
            # Check for NaN or Inf gradients
            if not torch.isfinite(torch.tensor(gradient_norm)):
                trainer.logger.log_metrics(
                    {"gradient/issue": 1}, step=trainer.global_step
                )
                print(f"[Warning] Non-finite gradient norm detected at step {trainer.global_step}.")
