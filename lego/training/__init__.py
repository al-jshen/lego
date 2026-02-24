import os
import time
from collections import OrderedDict, defaultdict
from math import ceil
from typing import Callable, Iterable, Literal, Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm.auto import tqdm

from lego.utils import CompilePolicy, Timer, to


class ModelOptState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_state_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, self.optimizer
        )
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict,
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )


class CheckpointManager:
    def __init__(self, async_save: bool = True, process_group=None):
        self.async_save = async_save
        if self.async_save:
            self.pg = process_group
            self.future = None

    def save(self, save_path, model, optimizer, scheduler, epoch, step):
        if self.pg is None:
            if dist.get_pg_count() > 0:
                self.pg = dist.new_group(backend="gloo")
            else:
                self.pg = dist.init_process_group(backend="gloo")

        # self.checkpointer.save(self.model, self.optimizer, self.scheduler, epoch, step)
        state_dict = {
            "model_opt": ModelOptState(model, optimizer),
            "scheduler": scheduler,
            "epoch": epoch,
            "step": step,
        }

        ckpt_path = os.path.join(save_path, f"epoch{epoch}_step{step}")
        os.makedirs(ckpt_path, exist_ok=True)

        storage_writer = FileSystemWriter(
            ckpt_path,
            thread_count=16,
            single_file_per_rank=True,
            sync_files=False,
        )

        if self.async_save and self.pg is not None:
            if self.future is not None:
                print("Waiting for previous checkpointing future to complete")
                try:
                    self.future.result()
                except Exception as e:
                    print(f"Previous checkpoint save failed: {e}")
                    # Continue with new checkpoint save attempt
                finally:
                    self.future = None

            self.future = dcp.async_save(
                state_dict,
                # checkpoint_id=os.path.join(save_path, f"epoch{epoch}_step{step}"),
                storage_writer=storage_writer,
                process_group=self.pg,
            )
        else:
            dcp.save(state_dict, storage_writer=storage_writer)

        if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
            print(f"Checkpoint saved at epoch {epoch}, step {step} to {save_path}")

    def load(self, load_path, model, optimizer, scheduler):
        state_dict = {
            "model_opt": ModelOptState(model, optimizer),
            "scheduler": scheduler,
            "epoch": 0,
            "step": 0,
        }
        dcp.load(
            state_dict=state_dict,
            # checkpoint_id=load_path,
            storage_reader=FileSystemReader(load_path),
        )
        # extra_state = torch.load(
        #     os.path.join(load_path, "extra_state.pt"),
        #     weights_only=False,
        # )
        if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
            print(f"Checkpoint loaded from {load_path}")

        return state_dict["epoch"], state_dict["step"]

    def cleanup(self):
        """Ensure all pending async operations complete before shutdown."""
        if self.future is not None:
            try:
                self.future.result()
            except Exception as e:
                print(f"Final checkpoint save failed: {e}")
            finally:
                self.future = None


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class fsdp_no_sync:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.model.set_requires_gradient_sync(False)
        return self.model

    def __exit__(self, exc_type, exc_value, traceback):
        self.model.set_requires_gradient_sync(True)

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


class Strategy:
    """Base class for strategies. Should be subclassed for specific strategies."""

    def __init__(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def wrap(self, model: nn.Module) -> nn.Module:
        """Wrap the model according to the strategy."""
        raise NotImplementedError("Subclasses should implement this method.")


class FSDPStrategy(Strategy):
    def __init__(
        self,
        sharding_strategy: Literal["full_shard", "hybrid_shard", "no_shard"],
        modules_to_wrap: Iterable[type[nn.Module]] = [],
        reshard_after_forward: Optional[bool] = None,
        cpu_offload: bool = False,
        mixed_precision_policy: Optional[MixedPrecisionPolicy] = None,
        **kwargs,
    ):
        self.modules_to_wrap = tuple(set(modules_to_wrap))

        self.sharding_strategy = sharding_strategy
        self.fsdp_kwargs = kwargs
        if cpu_offload:
            self.fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()
        if mixed_precision_policy is not None:
            self.fsdp_kwargs["mp_policy"] = mixed_precision_policy
        self.fsdp_kwargs["reshard_after_forward"] = reshard_after_forward

    def setup_device_mesh(
        self, world_size: int, local_world_size: int
    ) -> dist.device_mesh.DeviceMesh:
        """Setup device mesh for FSDP if needed."""

        if self.sharding_strategy == "hybrid_shard":
            mesh_shape = (world_size // local_world_size, local_world_size)
            mesh_dim_names = ("replicate", "shard")
        elif self.sharding_strategy == "full_shard":
            mesh_shape = (world_size,)
            mesh_dim_names = ("shard",)
        elif self.sharding_strategy == "no_shard":
            mesh_shape = (world_size, 1)
            mesh_dim_names = ("replicate", "shard")
        else:
            raise ValueError(f"Invalid sharding_strategy: {self.sharding_strategy}")

        device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_shape,
            mesh_dim_names=mesh_dim_names,
        )
        if dist.is_initialized():
            dist.barrier()

        return device_mesh

    def wrap(self, model: nn.Module, device_mesh=None, top_level=False):
        # If no mesh and not in a multi-rank process group, do not apply FSDP
        if device_mesh is None:
            if not (
                dist.is_available()
                and dist.is_initialized()
                and dist.get_world_size() > 1
            ):
                return model

        # apply fully_shard to model modules
        if len(self.modules_to_wrap) > 0:
            for name, module in model.named_children():
                if isinstance(module, self.modules_to_wrap):
                    fully_shard(module, mesh=device_mesh, **self.fsdp_kwargs)
                else:
                    # recurse in
                    self.wrap(module, device_mesh, top_level=False)

        if top_level:
            fully_shard(model, mesh=device_mesh, **self.fsdp_kwargs)


class ActivationCheckpointingStrategy(Strategy):
    def __init__(self, modules: bool | Iterable[type], reentrant: bool = False):
        if isinstance(modules, bool):
            if modules:
                self.activation_checkpointing = True
            else:
                self.activation_checkpointing = False
        elif isinstance(modules, Iterable):
            self.activation_checkpointing = tuple(set(modules))
        else:
            raise ValueError(
                f"modules must be bool or Iterable, got {type(modules)} instead"
            )

        self.checkpoint_impl = (
            CheckpointImpl.REENTRANT if reentrant else CheckpointImpl.NO_REENTRANT
        )

    def wrap(self, model: nn.Module) -> None:
        # activation checkpointing
        if (
            isinstance(self.activation_checkpointing, bool)
            and self.activation_checkpointing
        ):
            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(
                    m, checkpoint_impl=self.checkpoint_impl
                ),
            )
        elif isinstance(self.activation_checkpointing, Iterable):
            for module in model.modules():
                if isinstance(module, self.activation_checkpointing):
                    print(
                        f"Applying activation checkpointing to {module.__class__.__name__}"
                    )
                    apply_activation_checkpointing(
                        module,
                        checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(
                            m, checkpoint_impl=self.checkpoint_impl
                        ),
                    )
        else:
            raise ValueError(
                f"activation_checkpointing must be bool or Iterable, got {type(self.activation_checkpointing)} instead"
            )


class Logger:
    def __init__(self):
        raise NotImplementedError(
            "Logger is an abstract class and should not be instantiated directly."
        )

    def setup(self, model, cfg):
        """Setup the logger."""
        pass  # default: no op

    def log(self, metrics: dict, **kwargs):
        """Log metrics to the logger."""
        raise NotImplementedError("Subclasses should implement this method.")


class WandbLogger(Logger):
    def __init__(self, entity: str, project: str, watch: bool = True, **kwargs):
        self.entity = entity
        self.project = project
        self.kwargs = kwargs
        self.run = None
        self.watch = watch

    def setup(self, model, cfg):
        """Setup the wandb logger with the model."""
        run = wandb.init(entity=self.entity, project=self.project, **self.kwargs)
        self.run = run
        model_summary = get_model_summary(model)
        cfg = (
            OmegaConf.to_container(cfg, resolve=True)
            if isinstance(cfg, DictConfig)
            else cfg
        )
        cfg.update({f"model/{k}": v for k, v in model_summary.items()})
        wandb.config.update(cfg)
        if self.watch:
            wandb.watch(model, log="all")

    def log(self, metrics: dict, step: int = None):
        """Log metrics to Weights & Biases."""
        if not wandb.run:
            raise RuntimeError("WandbLogger requires an active wandb run.")
        wandb.log(metrics)


class TQDMLogger(Logger):
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.pbar = None

    def log(self, metrics: dict, step: int):
        """Log metrics to the tqdm progress bar."""
        if self.pbar is None:
            self.pbar = tqdm(
                total=self.total_steps, desc="Training Progress", unit="step"
            )
        self.pbar.update(step - self.pbar.n)
        self.pbar.set_description(f"Step {step}/{self.total_steps}")
        self.pbar.set_postfix(metrics)
        if step == self.total_steps - 1:
            self.pbar.close()


class CommandLineLogger(Logger):
    """Simple logger that prints metrics to stdout."""

    def __init__(self, log_format: str = "detailed", prefix: str = ""):
        """
        Initialize the command line logger.

        Args:
            log_format: "detailed" for multi-line format, "compact" for single-line format
            prefix: Optional prefix to add to all log messages
        """
        self.log_format = log_format
        self.prefix = prefix
        self.last_step = None

    def log(self, metrics: dict, step: int, **kwargs):
        """Log metrics to stdout."""
        # Extract step from metrics if not provided
        if step is None:
            step = metrics.get("global_step", metrics.get("step", "?"))

        # Format timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        if self.log_format == "compact":
            # Single line format
            metric_str = " | ".join(
                [
                    f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
                    for k, v in metrics.items()
                ]
            )
            if self.prefix:
                print(f"[{timestamp}] {self.prefix} | Step {step} | {metric_str}")
            else:
                print(f"[{timestamp}] Step {step} | {metric_str}")

        else:  # detailed format
            # Multi-line format with better readability
            if self.last_step != step or step == 0:
                print(f"\n{'=' * 60}")
                print(f"[{timestamp}] Step: {step}")
                if self.prefix:
                    print(f"Context: {self.prefix}")
                print("-" * 60)

                # Group metrics by prefix (e.g., train/, val/, etc.)
                grouped_metrics = {}
                ungrouped_metrics = {}

                for key, value in metrics.items():
                    if "/" in key:
                        prefix, name = key.split("/", 1)
                        if prefix not in grouped_metrics:
                            grouped_metrics[prefix] = {}
                        grouped_metrics[prefix][name] = value
                    else:
                        ungrouped_metrics[key] = value

                # Print grouped metrics
                for group, group_metrics in sorted(grouped_metrics.items()):
                    print(f"\n{group.capitalize()}:")
                    for name, value in sorted(group_metrics.items()):
                        if isinstance(value, float):
                            print(f"  {name:<25} {value:>15.6f}")
                        else:
                            print(f"  {name:<25} {value:>15}")

                # Print ungrouped metrics
                if ungrouped_metrics:
                    print("\nGeneral:")
                    for name, value in sorted(ungrouped_metrics.items()):
                        if name in ["step", "global_step"]:  # Skip redundant step info
                            continue
                        if isinstance(value, float):
                            print(f"  {name:<25} {value:>15.6f}")
                        else:
                            print(f"  {name:<25} {value:>15}")

                self.last_step = step


class Optimizer:
    def __init__(
        self,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        scheduler: Literal["cosine", "constant"] = "cosine",
        scheduler_max_steps: Optional[int] = None,
        warmup_steps: int = 0,
        min_lr: Optional[float] = 1e-6,
    ):
        self.lr = lr
        self.betas = betas
        if isinstance(self.betas, list):
            assert len(self.betas) == 2, "betas must be a tuple of two floats"
            self.betas = tuple(self.betas)
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.scheduler_max_steps = scheduler_max_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

    def setup(
        self, model
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:  # usually weight matrices
                decay_params.append(param)
            else:  # biases, norm weights
                no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = optim.AdamW(optim_groups, lr=self.lr, betas=self.betas)
        schedulers = []
        if self.warmup_steps > 0:
            schedulers.append(
                optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-6, total_iters=self.warmup_steps
                )
            )

        schedulers.append(
            optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_max_steps - self.warmup_steps,
                eta_min=self.min_lr,
            )
            if self.scheduler == "cosine"
            else optim.lr_scheduler.ConstantLR(optimizer, factor=1)
        )

        if self.scheduler == "cosine":
            schedulers.append(
                optim.lr_scheduler.ConstantLR(
                    optimizer,
                    factor=self.min_lr / self.lr,
                    total_iters=int(
                        1e9
                    ),  # set total_iters to big number, otherwise it reverts back to factor=1 after a bit
                )
            )

        milestones = []
        if self.warmup_steps > 0:
            milestones.append(self.warmup_steps)
        if self.scheduler == "cosine":
            milestones.append(self.scheduler_max_steps)
        if len(milestones) == 0:
            milestones = None

        if len(schedulers) > 1:
            assert milestones is not None

        lr_scheduler = (
            optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers,
                milestones=milestones,
            )
            if len(schedulers) > 1
            else schedulers[0]
        )
        return optimizer, lr_scheduler


def get_model_summary(model: nn.Module) -> dict:
    """Generate a summary of the model architecture."""
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    # Count parameters
    for param in model.parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        else:
            non_trainable_params += param_count

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
    }


def get_model_layer_info(model: nn.Module) -> OrderedDict:
    # Get layer information
    layers_info = OrderedDict()
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            param_count = sum(p.numel() for p in module.parameters())
            if param_count > 0:
                layers_info[name] = {
                    "type": module.__class__.__name__,
                    "params": param_count,
                    "output_shape": None,  # Could be computed with a forward pass
                }

    return layers_info


class Trainer:
    def __init__(
        self,
        model: Callable[[], nn.Module],
        optimizer: Optimizer,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        train_collate_fn: Optional[Callable] = None,
        val_collate_fn: Optional[Callable] = None,
        train_dataloader: Optional[Iterable] = None,
        val_dataloader: Optional[Iterable] = None,
        max_epochs: int = 1,
        max_steps: Optional[int] = None,
        log_every_n_steps: int | Iterable[int] = 50,
        gradient_accumulation_steps: int = 1,
        grad_clip_norm: Optional[float] = None,
        precision: str = "bf16",
        strategy: Optional[FSDPStrategy] = None,
        compile: bool
        | CompilePolicy = False,  # bool → whole model, set of types → selective
        ckpt_save_dir: str = "./checkpoints",
        ckpt_load_dir: Optional[str] = None,
        reset_steps: bool = False,
        ckpt_every_n_steps: Optional[int] = None,
        ckpt_every_n_epochs: int = 1,
        validate_every_n_steps: Optional[int] = None,
        validate_every_n_epochs: int = 1,
        async_checkpoint: bool = True,
        activation_checkpointing: Optional[ActivationCheckpointingStrategy] = None,
        logger: Optional[Logger | Iterable[Logger]] = None,
        drop_last: bool = True,
        batch_size: int = 32,
        num_workers: int = 8,
        shuffle: bool = True,
        pin_memory: bool = True,
        seed: int = 0,
        enable_timer: bool = False,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
    ):
        self.global_rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_collate_fn = train_collate_fn
        self.val_collate_fn = val_collate_fn

        if train_dataloader is not None:
            self._pre_built_train_loader = train_dataloader
            steps_per_epoch = len(train_dataloader)
        elif train_dataset is not None:
            self._pre_built_train_loader = None
            steps_per_epoch = int(
                ceil(
                    (len(self.train_dataset) - (1 if drop_last else 0))
                    / batch_size
                    / self.world_size
                )
            )
        else:
            raise ValueError(
                "Either train_dataset or train_dataloader must be provided"
            )
        self._pre_built_val_loader = val_dataloader

        # figure out how many epochs to train for
        if max_epochs is not None and max_steps is not None:
            raise ValueError("Only one of max_epochs or max_steps should be set.")
        if max_epochs is not None:
            max_steps = max_epochs * steps_per_epoch
        elif max_steps is not None:
            max_epochs = int(ceil(max_steps / steps_per_epoch))

        self.cfg = dict(
            max_epochs=max_epochs,
            max_steps=max_steps,
            log_every_n_steps=log_every_n_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            grad_clip_norm=grad_clip_norm,
            precision=precision,
            strategy=strategy,
            compile_cfg=compile,
            ckpt_save_dir=ckpt_save_dir,
            ckpt_load_dir=ckpt_load_dir,
            reset_steps=reset_steps,
            ckpt_every_n_steps=ckpt_every_n_steps,
            ckpt_every_n_epochs=ckpt_every_n_epochs,
            validate_every_n_steps=validate_every_n_steps,
            validate_every_n_epochs=validate_every_n_epochs,
            async_checkpoint=async_checkpoint,
            seed=seed,
            activation_checkpointing=activation_checkpointing,
            logger=logger,
            drop_last=drop_last,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            enable_timer=enable_timer,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
        )
        for k, v in self.cfg.items():
            setattr(self, k, v)

        if self.seed is not None:
            seed_everything(self.seed)

        if torch.cuda.is_available():
            # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(self.local_world_size)])
            print(
                f"[Rank {self.global_rank}] CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}"
            )
            self.device = f"cuda:{self.local_rank}"
            torch.cuda.set_device(self.device)
            print(f"[Rank {self.global_rank}] Device set to {self.device}")
        else:
            self.device = "cpu"

        # setup loggers
        if logger is None:
            self.logger = []
        elif isinstance(logger, Logger):
            self.logger = [logger]
        elif isinstance(logger, Iterable):
            for log in logger:
                if not isinstance(log, Logger):
                    raise ValueError(
                        f"All elements in logger list must be Logger, got {type(log)} instead"
                    )
            self.logger = list(logger)
        else:
            raise ValueError(
                f"logger must be Logger or list of Logger, got {type(logger)} instead"
            )

        if isinstance(self.log_every_n_steps, int):
            self.log_every_n_steps = [self.log_every_n_steps] * len(self.logger)
        elif isinstance(log_every_n_steps, Iterable):
            self.log_every_n_steps = list(log_every_n_steps)
            if len(self.log_every_n_steps) != len(self.logger):
                raise ValueError(
                    "log_every_n_steps must be a single int or a list with the same length as logger list"
                )
        else:
            raise ValueError(
                f"log_every_n_steps must be int or list, got {type(log_every_n_steps)} instead"
            )

        self.use_amp = self.precision != "fp32"
        dtype_map = dict(
            fp32=torch.float32,
            bf16=torch.bfloat16,
            fp16=torch.float16,
        )
        dtype = dtype_map[self.precision]
        self.autocast = torch.autocast(
            enabled=self.use_amp, dtype=dtype, device_type=self.device
        )

        self.train_loader = self._pre_built_train_loader
        self.val_loader = self._pre_built_val_loader

        # ====================
        # SETUP
        # ====================

        # move model to device
        self.model = self.model.to(self.device)

        # wrap model (fsdp, checkpointing) BEFORE compile
        self.device_mesh = self.distribute_model(self.model)

        if isinstance(compile, bool):
            if compile:
                self.model = torch.compile(self.model)
        elif isinstance(compile, CompilePolicy):
            self.model = compile(self.model)
        else:
            raise ValueError(
                f"compile must be bool or CompilePolicy, got {type(compile)} instead"
            )
        self.process_group = (
            self.device_mesh.get_group("shard") if self.device_mesh else None
        )

        self.base_optimizer = optimizer
        self.optimizer, self.scheduler = optimizer.setup(self.model)

        self.ckpt_manager = CheckpointManager(
            async_save=self.async_checkpoint
        )  # , process_group=self.process_group)

        # resume
        self.start_epoch = 0
        self.global_step = 0
        if self.ckpt_load_dir:
            epoch, step = self._load_checkpoint(self.ckpt_load_dir)
            if not self.reset_steps:
                self.start_epoch = epoch
                self.global_step = step
            else:
                self.optimizer, self.scheduler = self.base_optimizer.setup(self.model)
            if (
                not (dist.is_available() and dist.is_initialized())
                or dist.get_rank() == 0
            ):
                print(
                    f"Loaded model from checkpoint, starting from epoch {self.start_epoch}, step {self.global_step}"
                )

    @property
    def on_cuda(self):
        """Check if the model is on CUDA."""
        return torch.cuda.is_available() and self.device.startswith("cuda")

    # -------------------------------
    # Wrapping logic (DDP, FSDP, etc.) and activation checkpointing
    # -------------------------------
    def distribute_model(
        self, model: nn.Module, **kwargs
    ) -> Optional[dist.device_mesh.DeviceMesh]:
        if not self.strategy and self.world_size <= 1:
            print(
                f"[Rank {self.global_rank}] Running in single-GPU mode, no wrapping applied."
            )
            return None

        if self.strategy is None:
            print(
                f"[Rank {self.global_rank}] No strategy specified, using NO_SHARD FSDP strategy."
            )
            self.strategy = FSDPStrategy("no_shard", cpu_offload=False)

        assert isinstance(self.strategy, FSDPStrategy), (
            "strategy must be an instance of FSDPStrategy"
        )

        device_mesh = self.strategy.setup_device_mesh(
            self.world_size, self.local_world_size
        )

        print(
            f"[Rank {self.global_rank}] Using {self.strategy.sharding_strategy} strategy."
        )
        self.strategy.wrap(model, device_mesh=device_mesh, top_level=True)

        return device_mesh

    def apply_activation_checkpointing(self, model: nn.Module, **kwargs) -> nn.Module:
        if not self.activation_checkpointing:
            return

        if not isinstance(
            self.activation_checkpointing, ActivationCheckpointingStrategy
        ):
            raise ValueError(
                f"activation_checkpointing must be ActivationCheckpointingStrategy, got {type(self.activation_checkpointing)} instead"
            )

        print(
            f"[Rank {self.global_rank}] Applying activation checkpointing with modules: {self.activation_checkpointing.activation_checkpointing}"
        )
        self.activation_checkpointing.wrap(model, **kwargs)

    # -------------------------------
    # Checkpointing
    # -------------------------------
    def _save_checkpoint(self, epoch: int, step: int):
        # # self.checkpointer.save(self.model, self.optimizer, self.scheduler, epoch, step)
        # state_dict = {"app": Checkpointer(self.model, self.optimizer)}
        # dcp.async_save(state_dict, checkpoint_id=os.path.join(self.ckpt_save_dir, f"epoch{epoch}_step{step}"))
        # if torch.distributed.get_rank() == 0:
        #     # save scheduler state, epoch, and step
        #     extra_state = {
        #         "scheduler_state_dict": self.scheduler.state_dict(),
        #         "epoch": epoch,
        #         "global_step": step,
        #     }
        #     torch.save(extra_state, os.path.join(self.ckpt_save_dir, f"epoch{epoch}_step{step}", "extra_state.pt"))
        #     print(f"[Rank {self.global_rank}] Checkpoint saved at epoch {epoch}, step {step} to {self.ckpt_save_dir}")

        self.ckpt_manager.save(
            self.ckpt_save_dir, self.model, self.optimizer, self.scheduler, epoch, step
        )

    def _save_torch_checkpoint(self, epoch: int, step: int):
        """Save model weights as a standard torch checkpoint for single-GPU loading."""
        state_dict = get_model_state_dict(
            self.model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )

        if self.is_rank_zero:
            ckpt_path = os.path.join(
                self.ckpt_save_dir, f"epoch{epoch}_step{step}_torch.pt"
            )
            torch.save(state_dict, ckpt_path)
            print(f"Standard torch checkpoint saved to {ckpt_path}")

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _load_checkpoint(self, load_path: str) -> tuple[int, int]:
        # map_location = {"cuda:%d" % 0: "cuda:%d" % self.global_rank}
        # checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        # self.model.load_state_dict(checkpoint["model"])
        # self.optimizer.load_state_dict(checkpoint["optimizer"])
        # if self.scheduler and checkpoint["scheduler"]:
        #     self.scheduler.load_state_dict(checkpoint["scheduler"])
        # self.start_epoch = checkpoint["epoch"]
        # self.global_step = checkpoint["step"]
        # print(
        #     f"[Rank {self.global_rank}] Resumed from checkpoint {path} (epoch {self.start_epoch}, step {self.global_step})"
        # )
        #
        # self.checkpointer.load_model(self.model)
        # self.checkpointer.load_optim(self.model, self.optimizer)
        # epoch, step = self.checkpointer.load_extra(self.scheduler)
        # self.start_epoch = epoch
        # self.global_step = step
        #
        # state_dict = {"app": Checkpointer(self.model, self.optimizer)}
        # dcp.load(
        #     state_dict=state_dict,
        #     checkpoint_id=self.ckpt_load_dir,
        # )
        # if torch.distributed.get_rank() == 0:
        #     extra_state = torch.load(
        #         os.path.join(self.ckpt_load_dir, "extra_state.pt"),
        #         weights_only=False,
        #     )
        #     print(f"[Rank {self.global_rank}] Checkpoint loaded from {self.ckpt_load_dir}")
        # self.scheduler.load_state_dict(extra_state["scheduler_state_dict"])
        # self.start_epoch = extra_state["epoch"]
        # self.global_step = extra_state["global_step"]
        return self.ckpt_manager.load(
            load_path, self.model, self.optimizer, self.scheduler
        )

    def _prepare_dataloader(
        self, dataset: Optional[Dataset], split: Optional[str] = None
    ):
        if dataset is None:
            return None
        if self.world_size <= 1:
            sampler = None
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=self.shuffle,
                drop_last=self.drop_last,
            )

        collate_fn = None
        if split == "train":
            collate_fn = self.train_collate_fn
        elif split == "val":
            collate_fn = self.val_collate_fn

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None and self.shuffle),
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=collate_fn,
        )

        return loader

    def _set_epoch_on_samplers(self, epoch: int):
        for ldr in [self.train_loader, self.val_loader]:
            if (
                ldr is not None
                and hasattr(ldr, "sampler")
                and isinstance(ldr.sampler, DistributedSampler)
            ):
                ldr.sampler.set_epoch(epoch)

    def _log(self, log_name, loss, step: Optional[int] = None):
        for logger in self.logger:
            if not isinstance(loss, dict):
                logger.log({f"{log_name}/loss": loss}, step=step)
            else:
                log_dict = {f"{log_name}/{k}": v for k, v in loss.items()}
                logger.log(log_dict, step=step)

    def extract_loss(self, loss_aux):
        if not isinstance(loss_aux, dict):
            return loss_aux
        else:
            return loss_aux["loss"]

    def training_step(self, batch, step):
        if hasattr(self.model, "training_step"):
            loss_aux = self.model.training_step(batch, step)
        elif hasattr(self.model, "step"):
            loss_aux = self.model.step(batch, step)
        else:
            raise NotImplementedError(
                "Model must implement either `training_step` or `step` method for training."
            )

        # if self.is_rank_zero:
        #     self._log("train", loss_aux)

        # loss = self.extract_loss(loss_aux)
        return loss_aux

    def validation_step(self, batch, step):
        if hasattr(self.model, "validation_step"):
            loss_aux = self.model.validation_step(batch, step)
        elif hasattr(self.model, "step"):
            loss_aux = self.model.step(batch, step)
        else:
            raise NotImplementedError(
                "Model must implement either `validation_step` or `step` method for training."
            )

        # if self.is_rank_zero:
        #     self._log("val", loss_aux)

        # loss = self.extract_loss(loss_aux)
        return loss_aux

    @property
    def is_rank_zero(self):
        return self.global_rank == 0

    def print_model_summary(self, max_depth: int = 3):
        """Print a formatted summary of the model architecture."""
        summary = get_model_summary(self.model)
        # layers_info = get_model_layer_info(self.model)

        print(self.model)

        print("\n" + "=" * 80)
        print(" " * 30 + "MODEL SUMMARY")
        print("=" * 80)

        # Print parameter counts
        print(f"\nTotal Parameters:        {summary['total_params']:,}")
        print(f"Trainable Parameters:    {summary['trainable_params']:,}")
        print(f"Non-trainable Parameters: {summary['non_trainable_params']:,}")

        # Calculate model size in MB based on actual parameter dtype
        total_size_bytes = 0
        for param in self.model.parameters():
            # Get the size of one element in bytes based on dtype
            if param.dtype == torch.float32:
                element_size = 4
            elif param.dtype == torch.float16 or param.dtype == torch.bfloat16:
                element_size = 2
            elif param.dtype == torch.float64:
                element_size = 8
            else:
                element_size = 4  # Default fallback

            total_size_bytes += param.numel() * element_size

        param_size_mb = total_size_bytes / (1024 * 1024)
        print(f"Model Size (MB):         {param_size_mb:.2f}")

        # # Print layer summary (limited depth)
        # print("\n" + "-" * 80)
        # print(f"{'Layer (type)':<50} {'Param #':<15}")
        # print("=" * 80)
        #
        # for i, (name, info) in enumerate(layers_info.items()):
        #     if i >= 20:  # Limit number of layers shown
        #         remaining = len(layers_info) - i
        #         print(f"\n... and {remaining} more layers")
        #         break
        #
        #     # Truncate long names and respect max_depth
        #     depth = name.count(".")
        #     if depth <= max_depth:
        #         display_name = name if len(name) <= 45 else name[:42] + "..."
        #         print(f"{display_name:<50} {info['params']:>14,}")

        print("=" * 80)

    def print_training_configuration(self):
        """Print the training configuration."""
        print("\n" + "=" * 80)
        print(" " * 25 + "TRAINING CONFIGURATION")
        print("=" * 80)

        config_items = [
            (
                "General",
                [
                    ("Device", self.device),
                    ("Precision", self.precision),
                    ("Seed", self.seed),
                    ("Max Epochs", self.max_epochs),
                    ("Max Steps", self.max_steps),
                    ("World Size", self.world_size),
                ],
            ),
            (
                "Data",
                [
                    ("Batch size", self.batch_size),
                    (
                        "Effective batch size",
                        self.batch_size
                        * self.gradient_accumulation_steps
                        * self.world_size,
                    ),
                    (
                        "Training steps per epoch (no accum)",
                        f"{int(ceil(len(self.train_dataset) / self.batch_size / self.world_size)):,}"
                        if self.train_dataset
                        else (
                            f"{len(self.train_loader):,}"
                            if self.train_loader
                            else "N/A"
                        ),
                    ),
                    (
                        "Val steps per epoch (no accum)",
                        f"{int(ceil(len(self.val_dataset) / self.batch_size / self.world_size)):,}"
                        if self.val_dataset
                        else (
                            f"{len(self.val_loader):,}"
                            if self.val_loader
                            else "N/A"
                        ),
                    ),
                    ("Custom train collate fn", self.train_collate_fn is not None),
                    ("Custom val collate fn", self.val_collate_fn is not None),
                    ("Num Workers", self.num_workers),
                    ("Pin Memory", self.pin_memory),
                    ("Drop Last", self.drop_last),
                    ("Shuffle", self.shuffle),
                ],
            ),
            (
                "Optimization",
                [
                    ("Peak Learning Rate", self.base_optimizer.lr),
                    ("Min Learning Rate", self.base_optimizer.min_lr),
                    ("Weight Decay", self.base_optimizer.weight_decay),
                    ("Gradient Accumulation Steps", self.gradient_accumulation_steps),
                    (
                        "Gradient Clip Norm",
                        self.grad_clip_norm if self.grad_clip_norm else "None",
                    ),
                    ("Warmup Steps", self.base_optimizer.warmup_steps),
                    ("Scheduler", self.base_optimizer.scheduler),
                    (
                        "Scheduler max steps",
                        self.base_optimizer.scheduler_max_steps
                        if self.base_optimizer.scheduler_max_steps == "cosine"
                        else "N/A",
                    ),
                ],
            ),
            (
                "Strategy",
                [
                    (
                        "Type",
                        f"FSDP {self.strategy.sharding_strategy}"
                        if self.strategy
                        else "Single GPU",
                    ),
                    (
                        "CPU OffloadPolicy",
                        "Enabled"
                        if self.strategy
                        and self.strategy.fsdp_kwargs.get("offload_policy")
                        else "Disabled",
                    ),
                    (
                        "Reshard after forward",
                        "Enabled (ZeRO-3)"
                        if self.strategy
                        and self.strategy.fsdp_kwargs.get("reshard_after_forward")
                        else "Disabled (ZeRO-2)",
                    ),
                    (
                        "Activation Checkpointing",
                        "Enabled" if self.activation_checkpointing else "Disabled",
                    ),
                    ("torch.compile", "Enabled" if self.compile_cfg else "Disabled"),
                ],
            ),
            (
                "Checkpointing",
                [
                    ("Checkpoint Directory", self.ckpt_save_dir),
                    (
                        "Resumed From",
                        self.ckpt_load_dir if self.ckpt_load_dir else "None",
                    ),
                    (
                        "Save Every N Steps",
                        self.ckpt_every_n_steps
                        if self.ckpt_every_n_steps
                        else "Disabled",
                    ),
                    (
                        "Save Every N Epochs",
                        self.ckpt_every_n_epochs
                        if self.ckpt_every_n_epochs
                        else "Disabled",
                    ),
                    (
                        "Validate Every N Steps",
                        self.validate_every_n_steps
                        if (self.val_dataset or self.val_loader)
                        else "Disabled",
                    ),
                    (
                        "Validate Every N Epochs",
                        self.validate_every_n_epochs
                        if (self.val_dataset or self.val_loader)
                        else "Disabled",
                    ),
                ],
            ),
            (
                "Logging",
                [
                    ("Log Every N Steps", self.log_every_n_steps),
                    (
                        "Loggers",
                        ", ".join([logger.__class__.__name__ for logger in self.logger])
                        if self.logger
                        else "None",
                    ),
                    ("Enable Timer", self.enable_timer),
                ],
            ),
        ]

        for section_name, items in config_items:
            print(f"\n{section_name}:")
            print("-" * 40)
            for key, value in items:
                print(f"  {key:<35} {value}")

        # Print limits if set
        if self.limit_train_batches or self.limit_val_batches:
            print("\nData Limits:")
            print("-" * 40)
            if self.limit_train_batches:
                print(f"  Train Batches Limit:         {self.limit_train_batches}")
            if self.limit_val_batches:
                print(f"  Val Batches Limit:           {self.limit_val_batches}")

        print("=" * 80 + "\n")

    def set_run_cfg(self, run_cfg: dict):
        """Set the run configuration."""
        self.run_cfg = run_cfg

    def setup_loggers(self):
        if self.is_rank_zero:  # setup
            for logger in self.logger:
                logger.setup(self.model, self.run_cfg or self.cfg)

            # make unique checkpoint directories
            extra_path = None
            for logger in self.logger:
                if isinstance(logger, WandbLogger) and logger.run:
                    # if using wandb, use its run id to name the checkpoint folder
                    extra_path = logger.run.id
            if extra_path is None:
                # otherwise use timestamp (yyyy-mm-dd-hh-mm-ss)
                extra_path = time.strftime("%Y-%m-%d-%H-%M-%S")

            extra_path = [extra_path]
        else:
            extra_path = [None]

        # make sure all ranks get same save path
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(extra_path, src=0)
            dist.barrier()

        self.ckpt_save_dir = os.path.join(self.ckpt_save_dir, extra_path[0])

    # -------------------------------
    # Training loop
    # -------------------------------
    def train(self):
        scaler = torch.amp.GradScaler(self.device, enabled=self.precision == "fp16")

        self.setup_loggers()

        if self.is_rank_zero:  # setup
            self.print_model_summary()
            self.print_training_configuration()
            print("\nStarting training...\n")

        for epoch in range(self.start_epoch, self.max_epochs):
            self.model.train()
            if self.train_loader is None:
                self.train_loader = self._prepare_dataloader(
                    self.train_dataset, split="train"
                )
            self._set_epoch_on_samplers(epoch)

            if self.enable_timer:
                end_time = time.perf_counter()

            for step, batch in enumerate(self.train_loader):
                if (
                    self.limit_train_batches is not None
                    and step >= self.limit_train_batches
                ):
                    break

                if self.enable_timer:
                    start_time = time.perf_counter()

                # transfer to gpu
                with Timer(
                    enabled=self.enable_timer, cuda=self.on_cuda
                ) as transfer_timer:
                    batch = to(batch, self.device)

                with (
                    Timer(
                        enabled=self.enable_timer, cuda=self.on_cuda
                    ) as forward_timer,
                    self.autocast,
                ):
                    loss_aux = self.training_step(batch, step)
                    loss = self.extract_loss(loss_aux)
                    loss = (
                        loss / self.gradient_accumulation_steps
                    )  # scale loss for accumulation

                    if torch.isnan(loss):
                        err_msg = f"NaN loss detected at epoch {epoch}, step {step}."
                        if self.is_rank_zero:
                            print(err_msg)
                            # dump last batch of data, and also model state dict
                            torch.save(
                                dict(
                                    batch=batch,
                                    model_state_dict=self.model.state_dict(),
                                ),
                                f"nan-dump-epoch{epoch}-step{step}-rank{self.global_rank}.pt",
                            )
                            print(
                                f"Dumped batch and model state dict to nan-dump-epoch{epoch}-step{step}-rank{self.global_rank}.pt"
                            )
                        raise ValueError(err_msg)

                del batch

                with Timer(
                    enabled=self.enable_timer, cuda=self.on_cuda
                ) as backward_timer:
                    if (
                        (step + 1) % self.gradient_accumulation_steps != 0
                    ):  # just calculate gradients, no step
                        # avoid unnecessary gradient syncing
                        with fsdp_no_sync(self.model):
                            scaler.scale(loss).backward()
                    else:  # step
                        scaler.scale(loss).backward()

                        if self.grad_clip_norm is not None:
                            scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.grad_clip_norm
                            )

                        scaler.step(self.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        if self.scheduler:
                            self.scheduler.step()

                self.global_step += 1

                # logging
                if self.is_rank_zero and self.logger:
                    log_dict_cache = {}
                    for logger, log_every_n_steps in zip(
                        self.logger, self.log_every_n_steps
                    ):
                        if self.global_step % log_every_n_steps == 0:
                            if self.global_step not in log_dict_cache:
                                if not isinstance(loss_aux, dict):
                                    log_dict = {"train/loss": loss}
                                else:
                                    log_dict = {
                                        f"train/{k}": v for k, v in loss_aux.items()
                                    }
                                    log_dict["train/loss"] = loss

                                lr = self.scheduler.get_last_lr()[
                                    0
                                ]  # we have two param groups, but they get the same lr
                                log_dict.update(
                                    {
                                        "lr": lr,
                                        "global_step": self.global_step,
                                    }
                                )
                                log_dict["train/loss"] = (
                                    log_dict["train/loss"]
                                    * self.gradient_accumulation_steps
                                )

                                if self.enable_timer:
                                    log_dict.update(
                                        {
                                            "train/forward_time": forward_timer.elapsed,
                                            "train/backward_time": backward_timer.elapsed,
                                            "train/data_time": start_time - end_time,
                                            "train/transfer_time": transfer_timer.elapsed,
                                        }
                                    )
                                log_dict_cache[self.global_step] = log_dict

                            logger.log(
                                log_dict_cache[self.global_step], step=self.global_step
                            )

                # checkpoint per step
                if (
                    self.ckpt_every_n_steps
                    and self.global_step % self.ckpt_every_n_steps == 0
                ):
                    if self.is_rank_zero:
                        print(
                            f"Saving checkpoint at epoch {epoch}, step {self.global_step}"
                        )
                    self._save_checkpoint(epoch, self.global_step)

                if self.enable_timer:
                    end_time = time.perf_counter()

                # optional validation
                if (
                    (self.val_dataset is not None or self.val_loader is not None)
                    and self.validate_every_n_steps
                    and self.global_step % self.validate_every_n_steps == 0
                ):
                    if self.val_loader is None:
                        self.val_loader = self._prepare_dataloader(
                            self.val_dataset, split="val"
                        )
                    self._set_epoch_on_samplers(epoch)
                    if self.is_rank_zero:
                        print(
                            f"[Rank {self.global_rank}] Validating after step {self.global_step}"
                        )
                    loss_aux = self.validate()

                    if self.is_rank_zero and self.logger:
                        if not isinstance(loss_aux, dict):
                            log_dict = {"val/loss": loss}
                        else:
                            log_dict = {f"val/{k}": v for k, v in loss_aux.items()}

                        log_dict.update(
                            {
                                "epoch": epoch + 1,
                                "global_step": self.global_step,
                            }
                        )

                        for logger in self.logger:
                            logger.log(log_dict, step=self.global_step)

            # checkpoint per epoch
            if self.ckpt_every_n_epochs and (
                (epoch + 1) % self.ckpt_every_n_epochs == 0
            ):
                if self.is_rank_zero:
                    print(
                        f"[Rank {self.global_rank}] Saving checkpoint at epoch {epoch + 1}, step {self.global_step}"
                    )
                self._save_checkpoint(epoch + 1, self.global_step)

            # optional validation
            if (
                (self.val_dataset is not None or self.val_loader is not None)
                and self.validate_every_n_epochs
                and (epoch + 1) % self.validate_every_n_epochs == 0
            ):
                if self.val_loader is None:
                    self.val_loader = self._prepare_dataloader(
                        self.val_dataset, split="val"
                    )
                self._set_epoch_on_samplers(epoch)
                if self.is_rank_zero:
                    print(
                        f"[Rank {self.global_rank}] Validating after epoch {epoch + 1}"
                    )

                loss_aux = self.validate()

                if self.is_rank_zero and self.logger:
                    if not isinstance(loss_aux, dict):
                        log_dict = {"val/loss": loss}
                    else:
                        log_dict = {f"val/{k}": v for k, v in loss_aux.items()}

                    log_dict.update(
                        {
                            "epoch": epoch + 1,
                            "global_step": self.global_step,
                        }
                    )

                    for logger in self.logger:
                        logger.log(log_dict, step=self.global_step)

        # do one last checkpoint to make sure we have final model saved
        if self.is_rank_zero:
            print(
                f"[Rank {self.global_rank}] Saving final checkpoint at epoch {self.max_epochs}, step {self.global_step}"
            )
        self._save_checkpoint(self.max_epochs, self.global_step)

        # Ensure all pending async checkpoint operations complete before shutdown
        self.ckpt_manager.cleanup()

        # Convert final checkpoint to standard torch format for single-GPU loading
        self._save_torch_checkpoint(self.max_epochs, self.global_step)

        if dist.get_pg_count() > 0:
            dist.destroy_process_group()

    # -------------------------------
    # Validation
    # -------------------------------
    def validate(self):
        self.model.eval()
        val_loss = defaultdict(float)
        with torch.no_grad(), self.autocast:
            for step, batch in enumerate(self.val_loader):
                if (
                    self.limit_val_batches is not None
                    and step >= self.limit_val_batches
                ):
                    break
                # transfer to evaluation device
                batch = to(batch, self.device)
                loss_aux = self.validation_step(batch, step)
                if not isinstance(loss_aux, dict):
                    val_loss["loss"] += loss_aux.item()
                else:
                    for k, v in loss_aux.items():
                        val_loss[k] += v.item()
                # loss = self.extract_loss(loss_aux)
                # val_loss += loss.item()
        for k, v in val_loss.items():
            val_loss[k] = v / (step + 1)
        return val_loss
