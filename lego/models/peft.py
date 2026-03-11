from typing import Optional
import torch.nn as nn
from peft import LoraConfig, get_peft_model


def wrap_lora(
    model: nn.Module, rank: int = 8, alpha: Optional[int] = None, rs_lora: bool = True
) -> nn.Module:
    """Wrap a model with LoRA.
    
    Args:
        model: The model to wrap.
        rank: The rank of the LoRA.
        alpha: The alpha of the LoRA. Commonly set to 2 * rank.
        rs_lora: Whether to use Rank-Stabilized LoRA.
    """
    config = LoraConfig(
        r=rank,
        target_modules="all-linear",
        lora_alpha=alpha if alpha is not None else rank * 2,
        use_rs_lora=rs_lora,
    )
    return get_peft_model(model, config)