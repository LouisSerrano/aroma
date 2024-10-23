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
from aroma.utils.data.load_data import set_seed
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch

from aroma.encoder_decoder_ks import (
    AROMAEncoderDecoderKL,
)
from huggingface_hub import hf_hub_download
import h5py


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@hydra.main(config_path="config/", config_name="inr_KS.yaml")
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
    dataset_name = cfg.data.dataset_name
    ntrain = cfg.data.ntrain
    ntest = cfg.data.ntest
    sub_tr = cfg.data.sub_tr
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

    path_train = hf_hub_download(
        repo_id="phlippe/Kuramoto-Sivashinsky-1D",
        filename="KS_train_fixed_viscosity.h5",
        repo_type="dataset",
    )
    path_valid = hf_hub_download(
        repo_id="phlippe/Kuramoto-Sivashinsky-1D",
        filename="KS_valid_fixed_viscosity.h5",
        repo_type="dataset",
    )

    f_tr = h5py.File(path_train)
    coords_tr = f_tr["train"]["x"][()]
    x_tr = f_tr["train"]["pde_140-256"][()]

    f_te = h5py.File(path_valid)
    coords_te = f_te["valid"]["x"][()]
    x_te = f_te["valid"]["pde_640-256"][()]

    run.tags = (
            ("inr",)
            + (dataset_name,)
            + (f"sub={sub_tr}",)
        )

    # trainset coords of shape (N, Dx, Dy, input_dim, T)
    input_dim = 1
    # trainset images of shape (N, Dx, Dy, output_dim, T)
    output_dim = 1

    print("input_dim", input_dim)

    # total frames trainset
    ntrain = len(x_tr)
    # total frames testset
    ntest = len(x_te)

    trainset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_tr).float(),
        torch.from_numpy(coords_tr)[:, None, :].repeat(1, x_tr.shape[1], 1).float(),
    )
    testset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_te).float(),
        torch.from_numpy(coords_te)[:, None, :].repeat(1, x_te.shape[1], 1).float(),
    )

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

    num_channels=1

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
        weight_decay=1e-6,  # cfg.optim.weight_decay
    )
    kl_weight = cfg.inr.kl_weight

    total_steps = cfg.optim.epochs * len(train_loader)
    scheduler_lr = CosineAnnealingLR(
        optimizer_inr, T_max=total_steps, eta_min=1e-5
    )  # change to epochs other wise

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
    sample_posterior = False
    bottleneck_first = False

    wandb.log({"results_dir": str(RESULTS_DIR)}, step=epoch_start, commit=False)

    for step in range(epoch_start, epochs):
        rel_train_mse = 0
        rel_test_mse = 0
        fit_train_mse = 0
        kl_train = 0
        fit_test_mse = 0
        kl_test = 0
        step_show = step % 100 == 0
        step_show_last = step == epochs - 1

        for substep, (images, coords) in enumerate(train_loader):
            inr.train()
            images = images.to(device)
            # coords = coords.to(device)
            n_samples = images.shape[0]

            # Generate random indices for columns
            indices = torch.rand(n_samples, 140).argsort(dim=1).to(images.device)
            shuffled_tensor = images.gather(
                1, indices.unsqueeze(-1).expand(n_samples, 140, 256)
            )
            # images = shuffled_tensor[:, 0, :].cuda()[..., None]
            num_t = 1
            images = (
                shuffled_tensor[:, :num_t, :]
                .cuda()[..., None]
                .reshape(n_samples * num_t, 256, 1)
            )

            coords = torch.linspace(0, 1, 256) * 2 - 1
            coords = coords.cuda()[None, :, None].repeat(n_samples * num_t, 1, 1)
            target = images.clone()

            if step < 100:
                distance = torch.cdist(coords[:, ::4, :], coords, p=1)
                mask = (distance < (2 / 64.0)).float()
                mask_b = (
                    torch.bernoulli(
                        torch.full((n_samples, mask.shape[1], mask.shape[2]), 0.01)
                    )
                    .int()
                    .to(mask)
                )
                mask = torch.clip(mask_b + mask, 0, 1)
            else:
                mask = None

            out, kl_loss = inr(
                images,
                coords,
                mask=mask,
                sample_posterior=sample_posterior,
                bottleneck_first=bottleneck_first,
            )  # ae
            mse_loss = ((out - target) ** 2).mean()
            rel_loss = batch_mse_rel_fn(out, target).mean()

            loss = rel_loss + kl_weight * kl_loss

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
            for images, coords in test_loader:
                inr.eval()
                images = images.to(device)
                coords = coords.to(device)
                n_samples = images.shape[0]

                # Generate random indices for columns
                indices = torch.rand(n_samples, 640).argsort(dim=1).to(images.device)
                shuffled_tensor = images.gather(
                    1, indices.unsqueeze(-1).expand(n_samples, 640, 256)
                )
                # images = shuffled_tensor[:, 0, :].cuda()[..., None]
                num_t = 1
                images = (
                    shuffled_tensor[:, :num_t, :]
                    .cuda()[..., None]
                    .reshape(n_samples * num_t, 256, 1)
                )

                coords = torch.linspace(0, 1, 256) * 2 - 1
                coords = coords.cuda()[None, :, None].repeat(n_samples * num_t, 1, 1)
                target = images.clone()

                out, kl_loss = inr(
                    images,
                    coords,
                    sample_posterior=sample_posterior,
                    bottleneck_first=bottleneck_first,
                )  # ae
                mse_loss = ((out - target) ** 2).mean()

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

        if True in (step_show, step_show_last):
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
