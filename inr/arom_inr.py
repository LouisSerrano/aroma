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

from aroma.losses import batch_mse_rel_fn
from aroma.utils.data.dynamics_dataset import TemporalDataset
from aroma.utils.data.dynamics_dataset import rearrange as rearrange_data
from aroma.utils.data.load_data import get_dynamics_data, set_seed
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch

from aroma.encoder_decoder import (
    AROMAEncoderDecoderKL, dropout_seq
)
import random

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@hydra.main(config_path="config/", config_name="inr.yaml")
def main(cfg: DictConfig) -> None:

    # neceassary for some reason now
    torch.set_default_dtype(torch.float32)

    # data
    saved_checkpoint = cfg.wandb.saved_checkpoint
    if saved_checkpoint:
        entity = cfg.wandb.entity
        project = cfg.wandb.project
        run_id = cfg.wandb.id
        run_name = cfg.wandb.name
        checkpoint = torch.load(cfg.wandb.checkpoint_path)
        cfg_inr = checkpoint["cfg"]
    elif saved_checkpoint == False:
        # wandb
        entity = cfg.wandb.entity
        project = cfg.wandb.project
        run_id = cfg.wandb.id
        run_name = cfg.wandb.name

    # data
    data_dir = cfg.data.dir
    dataset_name = cfg.data.dataset_name
    ntrain = cfg.data.ntrain
    ntest = cfg.data.ntest
    sub_from = cfg.data.sub_from
    sub_tr = cfg.data.sub_tr
    sub_te = cfg.data.sub_te
    sub_from = cfg.data.sub_from
    seed = cfg.data.seed
    same_grid = cfg.data.same_grid
    seq_inter_len = cfg.data.seq_inter_len
    seq_extra_len = cfg.data.seq_extra_len

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

    (u_train, u_eval_extrapolation, u_test, grid_tr, grid_tr_extra, grid_te) = (
        get_dynamics_data(
            data_dir,
            dataset_name,
            ntrain,
            ntest,
            seq_inter_len=seq_inter_len,
            seq_extra_len=seq_extra_len,
            sub_from=sub_from,
            sub_tr=sub_tr,
            sub_te=sub_te,
            same_grid=same_grid,
        )
    )
    model_type = "inr"
    print(
        f"data: {dataset_name}, u_train: {u_train.shape}, u_eval_extrapolation: {u_eval_extrapolation.shape}, u_test: {u_test.shape}"
    )
    print(
        f"grid: grid_tr: {grid_tr.shape}, grid_tr_extra: {grid_tr_extra.shape}, grid_te: {grid_te.shape}"
    )
    print("same_grid : ", same_grid)
    run.tags = ("inr",) + (model_type,) + (dataset_name,) + (f"sub={sub_tr}",)
   
    trainset = TemporalDataset(
        u_train, grid_tr, dataset_name
    )
    testset = TemporalDataset(
        u_test, grid_te, dataset_name
    )

    # trainset coords of shape (N, Dx, Dy, input_dim, T)
    input_dim = trainset.input_dim
    # trainset images of shape (N, Dx, Dy, output_dim, T)
    output_dim = trainset.output_dim

    print("input_dim", input_dim)

    # transforms datasets shape into (N * T, Dx, Dy, C)
    trainset = rearrange_data(trainset, dataset_name)
    testset = rearrange_data(testset, dataset_name)

    # total frames trainset
    ntrain = len(trainset)
    # total frames testset
    ntest = len(testset)

    # create torch dataset
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size_val,
        shuffle=True,
        num_workers=2,
    )

    if dataset_name in [
        "mp-pde-burgers",
        "navier-stokes-1e-3",
        "navier-stokes-1e-4",
        "navier-stokes-1e-5",
    ]:
        use_rel_loss = True
    else:
        use_rel_loss = False

    num_channels = 2 if cfg.data.dataset_name == "shallow-water-dino" else 1

    if saved_checkpoint:
        cfg = cfg_inr

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
            include_pos_in_value=cfg.inr.include_pos_in_value
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
    total_steps = cfg.optim.epochs * len(train_loader)
    scheduler_lr = CosineAnnealingLR(
        optimizer_inr, T_max=total_steps, eta_min=1e-5
    ) 

    if saved_checkpoint:
        inr.load_state_dict(checkpoint["inr"])
        optimizer_inr.load_state_dict(checkpoint["optimizer_inr"])
        cfg = checkpoint["cfg"]
        print("cfg : ", cfg)
    elif saved_checkpoint == False:
        epoch_start = 0
        best_loss = np.inf

    epoch_start = 0
    best_loss = np.inf

    wandb.log({"results_dir": str(RESULTS_DIR)}, step=epoch_start, commit=False)
    sample_posterior = cfg.inr.sample_posterior

    use_masking=False

    for step in range(epoch_start, epochs):
        rel_train_mse = 0
        rel_test_mse = 0
        fit_train_mse = 0
        kl_train = 0
        fit_test_mse = 0
        kl_test = 0
        step_show = step % 100 == 0
        step_show_last = step == epochs - 1

        for substep, (images, coords, idx) in enumerate(train_loader):
            inr.train()
            images = images.to(device)
            coords = coords.to(device)
            n_samples = images.shape[0]

            images = images.reshape(n_samples, -1, num_channels)
            coords = coords.reshape(n_samples, -1, input_dim)
            if dataset_name != "shallow-water-dino":
                coords = coords * 2 - 1

            coords_target = coords.clone()
            target = images.clone()

            if use_masking:
                dropout = random.choice([0, 0.25, 0.5])
                images, coords = dropout_seq(images, coords, dropout=dropout)#cfg.inr.dropout_sequence)
                out, kl_loss = inr(images, coords, target_coords=coords_target, sample_posterior=sample_posterior)
            else:
                out, kl_loss = inr(images, coords, sample_posterior=sample_posterior)

            if use_rel_loss:
                mse_loss = batch_mse_rel_fn(out, target).mean()
            else:
                mse_loss = ((out - target) ** 2).mean()

            loss = mse_loss + kl_weight * kl_loss

            optimizer_inr.zero_grad()
            loss.backward()
            optimizer_inr.step()
            scheduler_lr.step()  ## warning
            fit_train_mse += mse_loss.item() * n_samples
            kl_train += kl_loss.item() * n_samples

        train_loss = fit_train_mse / ntrain
        kl_train = kl_train / ntrain


        if True in (step_show, step_show_last):
            # Convert train_latent into a torch.nn.Parameter
            fit_test_mse = 0
            for images, coords, idx in test_loader:
                inr.eval()
                images = images.to(device)
                coords = coords.to(device)
                n_samples = images.shape[0]
                images = images.reshape(n_samples, -1, num_channels)
                coords = coords.reshape(n_samples, -1, input_dim)
                if dataset_name != "shallow-water-dino":
                    coords = coords * 2 - 1
                out, kl_loss = inr(images, coords, sample_posterior=sample_posterior)

                if use_rel_loss:
                    mse_loss = batch_mse_rel_fn(out, images).mean()
                else:
                    mse_loss = ((out - images) ** 2).mean()

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
                    "grid_tr": grid_tr,
                    "grid_te": grid_te,
                },
                f"{RESULTS_DIR}/{run_name}.pt",
            )

    return best_loss

if __name__ == "__main__":
    main()
