import hydra
import torch
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    torch.set_float32_matmul_precision(cfg.get("matmul_precision", "high"))
    torch.backends.fp32_precision = "tf32"

    # more fine-grained control below
    # torch.backends.cuda.matmul.fp32_precision = "ieee" # ieee or tf32
    # torch.backends.cudnn.fp32_precision = "ieee"
    # torch.backends.cudnn.conv.fp32_precision = "tf32"
    # torch.backends.cudnn.rnn.fp32_precision = "tf32"

    trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.set_run_cfg(cfg)

    trainer.train()


if __name__ == "__main__":
    train()