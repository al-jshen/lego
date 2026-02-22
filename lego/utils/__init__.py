import os
import time
from collections.abc import Mapping, MutableMapping, Sequence
from functools import reduce
from typing import Iterable, Set, Tuple, Type

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from einops import rearrange
from omegaconf import DictConfig, ListConfig, OmegaConf


class Timer:
    """Context manager to time the execution of a block of code."""

    def __init__(self, enabled: bool = True, cuda: bool = False):
        self.cuda = cuda
        self.enabled = enabled
        if enabled and cuda:
            self.start_time = torch.cuda.Event(enable_timing=True)
            self.end_time = torch.cuda.Event(enable_timing=True)
        elif enabled:
            self.start_time = None
            self.elapsed = None

    def __enter__(self):
        if not self.enabled:
            return self

        if self.cuda:
            self.start_time.record()
        else:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled:
            return

        if self.cuda:
            self.end_time.record()
            torch.cuda.synchronize()
            elapsed = (
                self.start_time.elapsed_time(self.end_time) / 1000.0
            )  # Convert to seconds
        else:
            elapsed = time.perf_counter() - self.start_time

        self.elapsed = elapsed


class Compose:
    """Composes a series of functions into a single callable."""

    def __init__(self, *functions):
        self.functions = functions

    def __call__(self, x):
        for f in self.functions:
            x = f(x)
        return x

    def __repr__(self):
        return f"Compose({self.functions})"


def compose(*functions):
    """Composes a series of functions."""
    return reduce(lambda f, g: lambda x: g(f(x)), functions)


def and_reduce(*functions):
    """Composes a series of functions that return boolean values.

    Returns True if all functions return True, otherwise returns False.
    """
    return reduce(lambda f, g: lambda x: f(x) and g(x), functions, lambda x: True)


def or_reduce(*functions):
    """Composes a series of functions that return boolean values.

    Returns True if any function returns True, otherwise returns False.
    """
    return reduce(lambda f, g: lambda x: f(x) or g(x), functions, lambda x: False)


def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python
    types."""
    if isinstance(obj, (ListConfig, DictConfig)):
        return (
            {k: convert_to_regular_types(v) for k, v in obj.items()}
            if isinstance(obj, DictConfig)
            else list(obj)
        )
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj


def disabled_train(self, mode=True):
    return self


def compile_parts(model, types: Tuple[Type, ...], **kwargs):
    for n, module in model.named_children():
        if isinstance(module, types):
            compiled = torch.compile(module, **kwargs)
            to_print = True
            if dist.is_initialized() and dist.get_rank() != 0:
                to_print = False
            if to_print:
                print(f"Compiling {module.__class__.__name__} {n}...")
            setattr(model, n, compiled)
        else:
            if len(list(module.children())) > 0:
                compile_parts(module, types, **kwargs)


class CompilePolicy:
    def __init__(self, types: Set[Type], **kwargs):
        self.types = types
        self.kwargs = kwargs

    def __call__(self, model):
        compile_parts(model, tuple(self.types), **self.kwargs)
        return model


def module_wrap_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    module_classes: Iterable[type[nn.Module]],
) -> bool:
    return isinstance(module, tuple(module_classes))


def find_unused_params(model):
    for n, p in model.named_parameters():
        if p.requires_grad_ and p.grad is None:
            print(n)


def uncompile(model):
    """Go through a pytorch model and remove all the compiled functions For
    anything that is torch._dynamo.eval_frame.OptimizedModule and has _orig_mod
    attribute, replace it with the _orig_mod."""
    for name, module in model.named_children():
        if isinstance(module, torch._dynamo.eval_frame.OptimizedModule) and hasattr(
            module, "_orig_mod"
        ):
            setattr(model, name, module._orig_mod)
        uncompile(module)
    return model


def tuplefy(x: Iterable):
    x = tuple(x)
    if len(x) == 1:
        return x[0]
    return x


def nparams(model, trainable_only=False):
    return sum(
        [p.numel() for p in model.parameters() if not trainable_only or p.requires_grad]
    )


def patchify(x, patch_size):
    """Patchify an image.

    Input is of shape B, C, H, ...
    """
    B, C, *spatial_dims = x.shape
    assert len(spatial_dims) == len(patch_size)
    nd = len(spatial_dims)
    num_patches = tuple([spatial_dims[i] // patch_size[i] for i in range(nd)])
    expansion = (1,) * (3 - nd)
    num_patches = num_patches + expansion
    patch_size = patch_size + expansion
    for _ in range(len(expansion)):
        x = x.unsqueeze(-1)
    x = rearrange(
        x,  #
        "b c (num_patch_x patch_size_x) (num_patch_y patch_size_y) (num_patch_z patch_size_z) -> b (num_patch_x num_patch_y num_patch_z) (patch_size_x patch_size_y patch_size_z c)",  # noqa: E501
        num_patch_x=num_patches[0],
        num_patch_y=num_patches[1],
        num_patch_z=num_patches[2],
        patch_size_x=patch_size[0],
        patch_size_y=patch_size[1],
        patch_size_z=patch_size[2],
        c=C,
    )
    return x


def unpatchify(x, patch_size, original_size=None):
    """Unpatchify.

    Input is of shape B, N, C
    """
    B, N, C = x.shape
    nd = len(patch_size)
    if original_size is None:
        patches_per_dim = int(N ** (1 / nd))
        original_size = tuple([ps * patches_per_dim for ps in patch_size])
    assert nd == len(original_size)
    expansion = (1,) * (3 - nd)
    patch_size = tuple(patch_size) + expansion
    original_size = tuple(original_size) + expansion
    num_patches = tuple([original_size[i] // patch_size[i] for i in range(3)])
    x = rearrange(
        x,
        "b (num_patch_x num_patch_y num_patch_z) (patch_size_x patch_size_y patch_size_z c) -> b c (num_patch_x patch_size_x) (num_patch_y patch_size_y) (num_patch_z patch_size_z)",  # noqa: E501
        num_patch_x=num_patches[0],
        num_patch_y=num_patches[1],
        num_patch_z=num_patches[2],
        patch_size_x=patch_size[0],
        patch_size_y=patch_size[1],
        patch_size_z=patch_size[2],
    )
    for _ in range(len(expansion)):
        x = x.squeeze(-1)
    return x


def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def linear_init(layer):
    nn.init.trunc_normal_(layer.weight, mean=0.0, std=0.02)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def default_init(module):
    """Reasonable default initialization for common layers."""
    if isinstance(
        module,
        (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.Embedding,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        ),
    ):
        return linear_init(module)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)
        return module
    else:
        return module


def on_channel_first(fn, x, *args, **kwargs):
    """Apply a function on a channel-first tensor, where the function expects
    to operate on a channel-last tensor."""
    x = rearrange(x, "b c ... -> b ... c")
    x = fn(x, *args, **kwargs)
    x = rearrange(x, "b ... c -> b c ...").contiguous()
    return x


def dropout_nd(x, p=0.5, training=True, inplace=False):
    """N-dimensional dropout."""
    return getattr(F, f"dropout{x.ndim - 1}d")(
        x, p, training, inplace
    )  # -1 for batch dim


def to(x, *args, **kwargs):
    """Like `torch.Tensor.to` but works with nested structures.

    Takes the same arguments as `torch.Tensor.to`.
    """
    return torch.utils._pytree.tree_map(
        lambda t: t.to(*args, **kwargs) if isinstance(t, torch.Tensor) else t, x
    )


def np_to_torch(x, *args, **kwargs):
    """Like `torch.Tensor.to` but works with nested structures.

    Takes the same arguments as `torch.Tensor.to`.
    """
    return torch.utils._pytree.tree_map(
        lambda t: torch.from_numpy(t).to(*args, **kwargs)
        if isinstance(t, np.ndarray)
        else t,
        x,
    )


def torch_to_np(x):
    """Convert torch tensors to numpy arrays in nested structures."""
    return torch.utils._pytree.tree_map(
        lambda t: t.cpu().numpy() if isinstance(t, torch.Tensor) else t, x
    )


def get_wandb_run(run_path: str):
    """Gets a W&B run from the run path (entity/project/run_id)"""
    wandb.login()
    wandb_api = wandb.Api()
    run = wandb_api.run(run_path)
    return run


def load_from_wandb(
    run_path: str, extra_config: dict = {}, try_load_checkpoint: bool = True
) -> dict:
    """Load a config from W&B run."""

    run = get_wandb_run(run_path)
    config = run.config
    if "strategy" in config["trainer"]:  # and config["trainer"]["strategy"] == "ddp":
        del config["trainer"]["strategy"]
    config = OmegaConf.merge(config, extra_config)
    instantiated_config = hydra.utils.instantiate(config)

    if try_load_checkpoint:
        try:
            ckpt_path = os.path.join(
                config["trainer"]["logger"]["save_dir"],
                run.project,
                run.id,
                "checkpoints",
                "last.ckpt",
            )
            ckpt = torch.load(
                ckpt_path,
                map_location=instantiated_config.model.device,
                weights_only=False,
            )
            instantiated_config.model.load_state_dict(ckpt["state_dict"], strict=False)
            print("Loaded checkpoint successfully!")
        except Exception as e:
            print("Could not load checkpoint!", e)
    return instantiated_config


def index_collated(collated, indices):
    def _index(elem):
        if isinstance(elem, (torch.Tensor, np.ndarray)):
            return elem[indices]
        elif isinstance(elem, Sequence) and not isinstance(elem, (str, bytes)):
            return np.array(elem)[indices]
        elif isinstance(elem, Mapping):
            return {key: _index(value) for key, value in elem.items()}
        else:
            return elem

    return torch.utils._pytree.tree_map(_index, collated)


def apply_to_collated(collated, fn):
    def _apply(elem):
        if isinstance(elem, (torch.Tensor, np.ndarray)):
            return fn(elem)
        elif isinstance(elem, Sequence) and not isinstance(elem, (str, bytes)):
            return type(elem)(_apply(e) for e in elem)
        elif isinstance(elem, Mapping):
            return {key: _apply(value) for key, value in elem.items()}
        else:
            return elem

    return torch.utils._pytree.tree_map(_apply, collated)


def collate_concat(batch):
    """Custom collate function that concatenates tensors using torch.concat
    while preserving nested dictionary and list structures.

    Args:
        batch (list): A batch of data samples (tensors, dictionaries, or lists).

    Returns:
        Collated batch with the same structure as the input but tensors concatenated.
    """
    elem = batch[0]

    if isinstance(elem, torch.Tensor):
        # Concatenate tensors along the first dimension
        return torch.concat(batch, dim=0)

    elif isinstance(elem, np.ndarray):
        # Concatenate tensors along the first dimension
        return np.concatenate(batch, axis=0)

    elif isinstance(elem, Mapping):
        # Recursively apply to dictionary values
        return {key: collate_concat([d[key] for d in batch]) for key in elem}

    elif isinstance(elem, Sequence) and not isinstance(elem, (str, bytes)):
        # Recursively apply to list/tuple elements
        return type(elem)(collate_concat(samples) for samples in zip(*batch))

    else:
        # For other types, return as is (default behavior)
        return batch


def flatten_dict(dictionary, parent_key="", separator="_"):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def collate_incomplete(list_of_dicts):
    """Like `default_collate`, but allows for batches that are missing some
    keys."""
    list_of_dicts = [d for d in list_of_dicts if d is not None]
    if not list_of_dicts:
        return {}
    keys = set().union(*(d.keys() for d in list_of_dicts))
    collated = {k: [] for k in keys}
    for d in list_of_dicts:
        for k in keys:
            collated[k].append(d.get(k, None))
    for k, v in collated.items():
        if any(isinstance(i, torch.Tensor) for i in v):
            assert not any(i is None for i in v), (
                "Cannot collate tensors with None values."
            )
            collated[k] = torch.stack(v, dim=0)
        elif any(isinstance(i, np.ndarray) for i in v):
            assert not any(i is None for i in v), (
                "Cannot collate numpy arrays with None values."
            )
            try:
                collated[k] = np.stack(v, axis=0)
            except ValueError as e:
                # print shapes
                shapes = [i.shape for i in v if isinstance(i, np.ndarray)]
                raise ValueError(
                    f"Cannot stack numpy arrays for key {k} with different shapes: {shapes}"
                ) from e
        elif all(i is None for i in v):
            collated[k] = None
        elif any(isinstance(i, (np.floating, np.integer)) for i in v) and not any(
            i is None for i in v
        ):
            collated[k] = np.array(collated[k])
        else:
            pass
            # collated[k] = v
    return collated


def module_wrap_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    module_classes: Iterable[type[nn.Module]],
) -> bool:
    return isinstance(module, tuple(module_classes))


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    return model


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True
    return model


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)
