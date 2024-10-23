import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
print(sys.executable)
import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from aroma.utils.data.load_data import set_seed
from aroma.utils.data.graph_dataset import (
    CylinderFlowDataset,
    AirfoilFlowDataset,
    PadCollate,
)
from aroma.encoder_decoder import (
    AROMAEncoderDecoderKL,
    dropout_seq,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import random


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def linear_scheduler(start, end, num_steps):
    delta = (end - start) / num_steps
    return [start + i * delta for i in range(num_steps)]


@hydra.main(config_path="config/", config_name="inr_cylinder.yaml")
def main(cfg: DictConfig) -> None:

    # neceassary for some reason now
    torch.set_default_dtype(torch.float32)  # try amp later

    # data
    saved_checkpoint = cfg.wandb.saved_checkpoint
    if saved_checkpoint:
        entity = cfg.wandb.entity
        project = cfg.wandb.project
        run_id = cfg.wandb.id
        run_name = cfg.wandb.name
        checkpoint = torch.load(cfg.wandb.checkpoint_path)
        cfg = checkpoint["cfg"]
    elif saved_checkpoint == False:
        # wandb
        entity = cfg.wandb.entity
        project = cfg.wandb.project
        run_id = cfg.wandb.id
        run_name = cfg.wandb.name

    # data
    dataset_name = cfg.data.dataset_name
    ntrain = cfg.data.ntrain
    ntest = cfg.data.ntest
    seed = cfg.data.seed

    # optim
    batch_size = cfg.optim.batch_size
    batch_size_val = (
        batch_size if cfg.optim.batch_size_val == None else cfg.optim.batch_size_val
    )
    lr_inr = cfg.optim.lr_inr
    epochs = cfg.optim.epochs

    # wandb
    run_dir = (
        os.path.join(os.getenv("WANDB_DIR"), f"wandb/{cfg.wandb.dir}/{dataset_name}")
        if cfg.wandb.dir is not None
        else None
    )
    device = torch.device("cuda")
    print("run dir given", run_dir)

    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        id=run_id,
        dir=run_dir,
        resume="allow",
    )

    wandb.config.update(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    run_name = wandb.run.name
    RESULTS_DIR = Path(os.getenv("WANDB_DIR")) / dataset_name / "inr"

    os.makedirs(str(RESULTS_DIR), exist_ok=True)

    set_seed(seed)

    if cfg.data.dataset_name == "cylinder-flow":
        trainset = CylinderFlowDataset(
            path=f"{cfg.data.dir}/meshgraphnet/cylinder_flow",
            split="train",
            noise=0.02,
            task="sliced",
            mode="random_t",
        )

        valset = CylinderFlowDataset(
            path=f"{cfg.data.dir}/meshgraphnet/cylinder_flow",
            split="valid",
            noise=0.02,
            task="sliced",
            mode="random_t",
        )

    elif cfg.data.dataset_name == "airfoil-flow":
        trainset = AirfoilFlowDataset(
            path=f"{cfg.data.dir}/meshgraphnet/airfoil",
            split="train",
            noise=0.02,
            task="sliced",
            mode="random_t",
        )

        valset = AirfoilFlowDataset(
            path=f"{cfg.data.dir}/meshgraphnet/airfoil",
            split="valid",
            noise=0.02,
            task="sliced",
            mode="random_t",
        )

    else:
        raise NotImplementedError

    # total frames trainset
    ntrain = len(trainset)
    # total frames testset
    ntest = len(valset)

    pad_collate = PadCollate()

    # create torch dataset
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=pad_collate,
    )
    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=pad_collate,
    )
    num_channels = 3 if cfg.data.dataset_name == "cylinder-flow" else 4
    input_dim = 2

    inr = (
        AROMAEncoderDecoderKL(
            num_channels=num_channels,
            input_dim=input_dim,
            num_self_attentions=cfg.inr.num_self_attentions,
            num_latents=cfg.inr.num_latents,
            hidden_dim=cfg.inr.hidden_dim,  # 256
            latent_dim=cfg.inr.latent_dim,  # latent_dim=8
            mlp_feature_dim=cfg.inr.mlp_feature_dim,
            dim=cfg.inr.dim,
            latent_heads=cfg.inr.latent_heads,
            latent_dim_head=cfg.inr.latent_dim_head,
            cross_heads=cfg.inr.cross_heads,
            cross_dim_head=cfg.inr.cross_dim_head,
            scales=cfg.inr.frequencies,
            num_freq=cfg.inr.num_freq,
            depth_inr=cfg.inr.depth_inr,
            bottleneck_index=cfg.inr.bottleneck_index,
            encode_geo=cfg.inr.encode_geo,
            max_pos_encoding_freq=cfg.inr.max_pos_encoding_freq,
        )
        .cuda()
        .float()
    )
    print("num params", count_parameters(inr))

    max_lr = lr_inr
    optimizer_inr = torch.optim.AdamW(
        [
            {"params": inr.parameters(), "lr": max_lr},
        ],
        lr=max_lr,
        weight_decay=cfg.optim.weight_decay,
    )
    kl_weight = cfg.inr.kl_weight

    scheduler_lr = CosineAnnealingLR(optimizer_inr, T_max=epochs, eta_min=1e-5)

    if saved_checkpoint:
        inr.load_state_dict(checkpoint["inr"])
        optimizer_inr.load_state_dict(checkpoint["optimizer_inr"])
        epoch_start = checkpoint["epoch"]
        cfg = checkpoint["cfg"]
        print("cfg : ", cfg)
    elif saved_checkpoint == False:
        epoch_start = 0
        best_loss = np.inf

    wandb.log({"results_dir": str(RESULTS_DIR)}, step=epoch_start, commit=False)

    for step in range(epoch_start, epochs):
        fit_train_mse = 0
        kl_train = 0
        fit_test_mse = 0
        kl_test = 0
        step_show = step % 100 == 0
        step_show_last = step == epochs - 1

        for substep, (coords, images, mask, idx) in enumerate(train_loader):
            inr.train()
            coords = coords.to(device).squeeze(
                -1
            )  # remove the temporal as we are in module random_t
            images = images.to(device).squeeze(
                -1
            )  # remove the temporal as we are in module random_t
            mask = mask.to(device)
            coords_target = coords.clone()
            target = images.clone()
            mask_target = mask.clone()

            dropout = random.choice([0, 0.25, 0.5, 0.75])
            images, coords, mask = dropout_seq(images, coords, mask, dropout=dropout)
            n_samples = images.shape[0]

            # try later
            # images, coords = dropout_seq(images, coords, dropout=cfg.inr.dropout_sequence)

            out, kl_loss = inr(images, coords, mask, target_coords=coords_target)
            mse_loss = (((out - target) ** 2) * mask_target[..., None]).sum() / (
                mask_target[..., None].sum() * num_channels
            )

            loss = mse_loss + kl_weight * kl_loss

            optimizer_inr.zero_grad()
            loss.backward()
            optimizer_inr.step()

            fit_train_mse += mse_loss.item() * n_samples
            kl_train += kl_loss.item() * n_samples

        train_loss = fit_train_mse / ntrain
        kl_train = kl_train / ntrain
        scheduler_lr.step()

        if True in (step_show, step_show_last):
            # Convert train_latent into a torch.nn.Parameter
            fit_test_mse = 0
            for substep, (coords, images, mask, idx) in enumerate(val_loader):
                inr.eval()
                coords = coords.to(device).squeeze(
                    -1
                )  # remove the temporal as we are in module random_t
                images = images.to(device).squeeze(
                    -1
                )  # remove the temporal as we are in module random_t
                mask = mask.to(device)
                n_samples = images.shape[0]

                # images, coords = dropout_seq(images, coords, dropout=cfg.inr.dropout_sequence)

                with torch.no_grad():
                    out, kl_loss = inr(images, coords, mask)
                mse_loss = (((out - images) ** 2) * mask[..., None]).sum() / (
                    mask[..., None].sum() * num_channels
                )

                fit_test_mse += mse_loss.item() * n_samples
                kl_test += kl_loss.item() * n_samples

            test_loss = fit_test_mse / ntest
            kl_test = kl_test / ntest

        if True in (step_show, step_show_last):
            wandb.log(
                {
                    "test_loss": test_loss,
                    "kl_test": kl_test,
                    "train_loss": train_loss,
                    "kl_train": kl_train,
                },
            )

        else:
            wandb.log(
                {"train_loss": train_loss, "kl_train": kl_train},
                step=step,
                commit=not step_show,
            )

        if train_loss < best_loss:
            best_loss = train_loss

            torch.save(
                {
                    "cfg": cfg,
                    "epoch": step,
                    "inr": inr.state_dict(),
                    "optimizer_inr": optimizer_inr.state_dict(),
                    "loss": best_loss,
                },
                f"{RESULTS_DIR}/{run_name}.pt",
            )

    return best_loss


if __name__ == "__main__":
    main()
