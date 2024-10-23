from aroma.utils.data.load_data import get_dynamics_data, set_seed
from aroma.utils.data.dynamics_dataset import KEY_TO_INDEX, TemporalDataset
from aroma.losses import batch_mse_rel_fn
from omegaconf import DictConfig, OmegaConf
import wandb
from torch import nn, einsum

import torch
import numpy as np
import hydra
import einops
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from aroma.encoder_decoder import (
    AROMAEncoderDecoderKL,
    DiagonalGaussianDistribution,
    dropout_seq,
)
from aroma.mlp import ResNet
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import random
from aroma.DIT_deterministic import DiT
from diffusers.schedulers import DDPMScheduler


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TokenDataset(Dataset):
    def __init__(
        self,
        dataset,
        mean,
        logvar,
        size_boost=1,
        use_polarized_mean=False,
        threshold=0.1,
    ):
        """
        Args:
            data (list): List of data items.
            labels (list): List of labels for the data items.
        """
        self.mean = mean
        self.logvar = logvar
        self.dataset = dataset
        self.size_boost = size_boost
        self.use_polarized_mean = use_polarized_mean
        self.threshold = threshold

    def __len__(self):
        return len(self.mean) * self.size_boost

    def __getitem__(self, idx):
        images, coords, idx = self.dataset.__getitem__(idx)

        mu = self.mean[idx]
        logvar = self.logvar[idx]

        sequence_size = mu.shape[0]
        T = mu.shape[-1]

        mu = einops.rearrange(mu, "n c t -> t n c")
        logvar = einops.rearrange(logvar, "n c t -> t n c")

        posterior = DiagonalGaussianDistribution(mu, logvar)

        codes = posterior.sample()

        codes = einops.rearrange(codes, "t n c -> n c t")
        logvar = einops.rearrange(logvar, "t n c -> n c t")
        mu = einops.rearrange(mu, "t n c -> n c t")

        if self.use_polarized_mean:
            mask = torch.exp(0.5 * logvar) > self.threshold
            mu = mu.clone()
            mu[mask] = 0  # for scaling experiments
            ## #mu[mask.all(-1)] = 0 # new

        return {
            "latent_codes": codes,
            "mean": mu,
            "logvar": logvar,
            "images": images,
            "coords": coords,
            "idx": idx,
            "sequence_size": sequence_size,
        }


@hydra.main(config_path="../config/", config_name="transformer.yaml")
def main(cfg: DictConfig) -> None:
    # neceassary for some reason now
    torch.set_default_dtype(torch.float32)

    # data
    data_dir = cfg.data.dir
    dataset_name = cfg.data.dataset_name
    ntrain = cfg.data.ntrain
    ntest = cfg.data.ntest
    sub_from = cfg.data.sub_from
    sub_tr = cfg.data.sub_tr
    sub_te = cfg.data.sub_te
    seed = cfg.data.seed
    same_grid = cfg.data.same_grid
    seq_inter_len = cfg.data.seq_inter_len
    seq_extra_len = cfg.data.seq_extra_len

    # optim
    batch_size = cfg.optim.batch_size
    epochs = cfg.optim.epochs

    # wandb
    entity = cfg.wandb.entity
    project = cfg.wandb.project
    run_id = cfg.wandb.id
    run_name = cfg.wandb.name
    run_dir = (
        os.path.join(os.getenv("WANDB_DIR"), f"wandb/{cfg.wandb.dir}")
        if cfg.wandb.dir is not None
        else None
    )
    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        id=run_id,
        dir=None,
    )

    wandb.config.update(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    run_name = wandb.run.name
    model_dir = Path(os.getenv("WANDB_DIR")) / dataset_name / "transformers"

    set_seed(seed)

    (u_train, u_train_eval, u_test, grid_tr, grid_tr_extra, grid_te) = (
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
    print(
        f"data: {dataset_name}, u_train: {u_train.shape}, u_train_eval: {u_train_eval.shape}, u_test: {u_test.shape}"
    )
    print(
        f"grid: grid_tr: {grid_tr.shape}, grid_tr_extra: {grid_tr_extra.shape}, grid_te: {grid_te.shape}"
    )

    run.tags = ("mlp",) + (dataset_name,) + (f"sub={sub_tr}",)

    trainset = TemporalDataset(u_train, grid_tr, dataset_name)
    trainset_extra = TemporalDataset(
        u_train_eval, grid_tr_extra, dataset_name
    )
    testset = TemporalDataset(
        u_test, grid_te, dataset_name
    )

    # total frames trainset
    ntrain = len(trainset)

    # total frames testset
    ntest = len(testset)

    # sequence length
    T_train = u_train.shape[-1]
    T_test = u_test.shape[-1]

    # trainset coords of shape (N, Dx, Dy, input_dim, T)
    input_dim = grid_tr.shape[-2]
    # trainset images of shape (N, Dx, Dy, output_dim, T)

    saved_checkpoint = cfg.wandb.saved_checkpoint
    if saved_checkpoint:
        entity = cfg.wandb.entity
        project = cfg.wandb.project
        run_id = cfg.wandb.id
        # run_name = cfg.wandb.name
        checkpoint = torch.load(cfg.wandb.checkpoint_path)
        cfg_inr = checkpoint["cfg"]
        # print("cfg_inr", cfg_inr)
    elif saved_checkpoint == False:
        # wandb
        entity = cfg.wandb.entity
        project = cfg.wandb.project
        run_id = cfg.wandb.id
        # run_name = cfg.wandb.name

    num_channels = 2 if cfg.data.dataset_name == "shallow-water-dino" else 1
    
    inr = (
        AROMAEncoderDecoderKL(
            num_channels=num_channels,
            input_dim=input_dim,
            num_self_attentions=cfg_inr.inr.num_self_attentions,
            num_latents=cfg_inr.inr.num_latents,
            hidden_dim=cfg_inr.inr.hidden_dim,  # 256
            latent_dim=cfg_inr.inr.latent_dim,  # latent_dim=8
            mlp_feature_dim=cfg_inr.inr.mlp_feature_dim,
            dim=cfg_inr.inr.dim,
            latent_heads=cfg_inr.inr.latent_heads,
            latent_dim_head=cfg_inr.inr.latent_dim_head,
            cross_heads=cfg_inr.inr.cross_heads,
            cross_dim_head=cfg_inr.inr.cross_dim_head,
            scales=cfg_inr.inr.frequencies,
            num_freq=cfg_inr.inr.num_freq,
            depth_inr=cfg_inr.inr.depth_inr,
            bottleneck_index=cfg_inr.inr.bottleneck_index,
            encode_geo=cfg_inr.inr.encode_geo,
            max_pos_encoding_freq=cfg_inr.inr.max_pos_encoding_freq,
        )
        .cuda()
        .float()
    )

    inr.load_state_dict(checkpoint["inr"])
    inr.eval()

    epoch_start = checkpoint["epoch"]
    best_loss = np.inf  # checkpoint['loss']
    cfg_inr = checkpoint["cfg"]
    print("cfg inr : ", cfg_inr)

    if dataset_name in [
        "navier-stokes-dino",
        "navier-stokes-1e-4",
        "navier-stokes-1e-5",
        "shallow-water-dino",
    ]:
        batch_size_tmp = 2
    else:
        batch_size_tmp = 16

    # create torch dataset
    train_loader_tmp = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size_tmp,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    train_extra_loader_tmp = torch.utils.data.DataLoader(
        trainset_extra,
        batch_size=batch_size_tmp,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    test_loader_tmp = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size_tmp,
        shuffle=False,
        num_workers=1,
    )

    num_latents = cfg_inr.inr.num_latents
    latent_dim = cfg_inr.inr.latent_dim  # latent_dim usually

    mu_train = torch.zeros(ntrain, num_latents, latent_dim, T_train)
    logvar_train = torch.zeros(ntrain, num_latents, latent_dim, T_train)

    mu_train_extra = torch.zeros(ntrain, num_latents, latent_dim, T_test)
    logvar_train_extra = torch.zeros(ntrain, num_latents, latent_dim, T_test)

    mu_test = torch.zeros(ntest, num_latents, latent_dim, T_test)
    logvar_test = torch.zeros(ntest, num_latents, latent_dim, T_test)

    print("before encoding")

    for substep, (images, coords, idx) in enumerate(train_loader_tmp):
        images = images.cuda()
        coords = coords.cuda()
        n_samples = images.shape[0]
        T = images.shape[-1]

        images = einops.rearrange(images, "b ... c t -> (t b) (...) c")
        coords = einops.rearrange(coords, "b ... c t -> (t b) (...) c")
        if cfg.data.dataset_name != "shallow-water-dino":
            coords = coords * 2 - 1

        with torch.no_grad():
            features, logvar = inr.encode(images, coords)

        mean = einops.rearrange(features, "(t b) n c -> b n c t", t=T)
        logvar = einops.rearrange(logvar, "(t b) n c -> b n c t", t=T)

        mu_train[idx] = mean.cpu().detach()
        logvar_train[idx] = logvar.cpu().detach()
    print("done encoding train")

    for substep, (images, coords, idx) in enumerate(
        train_extra_loader_tmp
    ):
        images = images.cuda()
        coords = coords.cuda()
        n_samples = images.shape[0]
        T = images.shape[-1]

        images = einops.rearrange(images, "b ... c t -> (t b) (...) c")
        coords = einops.rearrange(coords, "b ... c t -> (t b) (...) c")
        if cfg.data.dataset_name != "shallow-water-dino":
            coords = coords * 2 - 1

        with torch.no_grad():
            features, logvar = inr.encode(images, coords)

        mean = einops.rearrange(features, "(t b) n c -> b n c t", t=T)
        logvar = einops.rearrange(logvar, "(t b) n c -> b n c t", t=T)

        mu_train_extra[idx] = mean.cpu().detach()
        logvar_train_extra[idx] = logvar.cpu().detach()

    print("done encoding train extra")

    for substep, (images, coords, idx) in enumerate(test_loader_tmp):
        images = images.cuda()
        coords = coords.cuda()
        n_samples = images.shape[0]
        T = images.shape[-1]

        images = einops.rearrange(images, "b ... c t -> (t b) (...) c")
        coords = einops.rearrange(coords, "b ... c t -> (t b) (...) c")
        if cfg.data.dataset_name != "shallow-water-dino":
            coords = coords * 2 - 1

        with torch.no_grad():
            features, logvar = inr.encode(images, coords)

        mean = einops.rearrange(features, "(t b) n c -> b n c t", t=T)
        logvar = einops.rearrange(logvar, "(t b) n c -> b n c t", t=T)

        mu_test[idx] = mean.cpu().detach()
        logvar_test[idx] = logvar.cpu().detach()

    print("done encoding test")

    train_dataset_tkn = TokenDataset(
        trainset, mu_train, logvar_train, threshold=cfg.optim.threshold
    )
    train_dataset_extra_tkn = TokenDataset(
        trainset_extra,
        mu_train_extra,
        logvar_train_extra,
        threshold=cfg.optim.threshold,
    )
    test_dataset_tkn = TokenDataset(
        testset, mu_test, logvar_test, threshold=cfg.optim.threshold
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset_tkn, batch_size=batch_size, shuffle=True, num_workers=2
    )

    train_loader_extra = torch.utils.data.DataLoader(
        train_dataset_extra_tkn, batch_size=batch_size, shuffle=True, num_workers=1
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset_tkn, batch_size=batch_size, shuffle=True, num_workers=2
    )

    model = ResNet(
        input_dim=latent_dim,
        hidden_dim=512,
        output_dim=latent_dim,
        depth=4,
        dropout=0.0,
        activation="swish",
    ).cuda()

    try:
        resume_training = cfg.wandb.resume_training
    except:
        resume_training = False
    if resume_training:
        checkpoint_model = torch.load(cfg.wandb.prev_checkpoint_path)
        model.load_state_dict(checkpoint_model["model"])

    print("num params", count_parameters(model))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )

    scheduler_lr = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    # scheduler_lr = StepLR(optimizer, step_size=epochs//5, gamma=0.5)

    num_refinement_steps = 3
    min_noise_std = cfg.optim.min_noise  # 2e-6
    betas = [
        min_noise_std ** (k / num_refinement_steps)
        for k in reversed(range(num_refinement_steps + 1))
    ]
    scheduler = DDPMScheduler(
        num_train_timesteps=num_refinement_steps + 1,
        trained_betas=betas,
        prediction_type="v_prediction",
        clip_sample=False,
    )
    time_multiplier = 1000 / num_refinement_steps
    difference_weight = 1 / T_test if cfg.data.dataset_name == "mp-pde-burgers" else 1.0
    predict_difference = True if cfg.data.dataset_name == "mp-pde-burgers" else False
    # predict_difference = False
    use_statistics = True

    print("use statistics", use_statistics)

    for step in range(epochs):
        step_show = step % (epochs // 10) == 0
        step_show_last = step == epochs - 1

        pred_train_mse = 0
        code_train_mse = 0

        for substep, batch in enumerate(train_loader):
            model.train()
            codes = batch["latent_codes"].cuda()
            mean_batch = batch["mean"].cuda()
            logvar_batch = batch["logvar"].cuda()
            n_samples = codes.shape[0]
            t_dim = codes.shape[-1]
            inp = codes[..., 0]
            mu = mean_batch[..., 0]
            logvar = logvar_batch[..., 0]

            z_pred = torch.zeros_like(codes)
            z_pred[..., 0] = inp

            # optimizer.zero_grad()

            start = 0

            for t in range(start, t_dim - 1):

                # if t==0:
                if use_statistics:
                    x = mean_batch[..., t]
                else:
                    x = codes[..., t]
                # else:
                #    x = pred.detach()

                if use_statistics:
                    if predict_difference:
                        y = (
                            mean_batch[..., t + 1] - mean_batch[..., t]
                        ) / difference_weight
                    else:
                        y = mean_batch[..., t + 1]
                else:
                    if predict_difference:
                        y = (codes[..., t + 1] - codes[..., t]) / difference_weight
                    else:
                        y = codes[..., t + 1]

                y_noised = torch.zeros_like(x).to(x)
                # x_in = torch.cat([images[..., t], y_noised], axis=-1)
                pred = model(x)
                target = y
                # print('pred', pred.shape, target.shape, x_in.shape)
                loss = (
                    (pred - target) ** 2
                ).mean()  # self.train_criterion(pred, target

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                code_train_mse += loss.item() * n_samples

        code_train_mse = code_train_mse / (ntrain * t_dim)

        if True in (step_show, step_show_last):
            pred_train_mse = pred_train_mse / (ntrain * t_dim)

        scheduler_lr.step()

        if True in (step_show, step_show_last):
            pred_test_mse = 0
            code_test_mse = 0
            mse = 0
            mse_in = 0
            mse_out = 0

            for substep, batch in enumerate(test_loader):
                model.eval()
                codes = batch["latent_codes"].cuda()
                mean_batch = batch["mean"].cuda()
                logvar_batch = batch["logvar"].cuda()
                n_samples = codes.shape[0]
                t_dim = codes.shape[-1]
                inp = codes[..., 0]
                mu = mean_batch[..., 0]
                logvar = logvar_batch[..., 0]

                z_pred = torch.zeros_like(codes)
                z_pred[..., 0] = inp

                start = 0

                with torch.no_grad():
                    if use_statistics:
                        x = mean_batch[..., start]
                        z_pred[..., start] = x

                    else:
                        x = codes[..., start]
                        z_pred[..., start] = x

                    with torch.no_grad():
                        if use_statistics:
                            x = mean_batch[..., 0]
                        else:
                            x = codes[..., 0]

                        for t in range(t_dim - 1):
                            y_noised = torch.zeros_like(x).to(
                                x
                            )  # , dtype=x.dtype, device=x.device
                            pred = model(x)

                            y = pred.detach()
                            if predict_difference:
                                y = y * difference_weight + x
                            z_pred[..., t + 1] = y.detach()
                            x = y

                            # code_test_mse += ((mean_batch[..., t+1] - pred)**2).mean().item() * n_samples
                            code_test_mse += (
                                (mean_batch[..., t + 1] - y) ** 2
                            ).mean().item() * n_samples

                pred = z_pred.detach()
                images = batch["images"].cuda()
                coords = batch["coords"].cuda()
                images = images.reshape(n_samples, -1, num_channels, t_dim)
                coords = coords.reshape(n_samples, -1, input_dim, t_dim)
                if cfg.data.dataset_name != "shallow-water-dino":
                    coords = coords * 2 - 1
                tot_pred = torch.zeros_like(images)

                for t in range(start, t_dim):
                    x = pred[..., t]
                    c = coords[..., t]
                    y = images[..., t]

                    with torch.no_grad():
                        u_pred = inr.decode(x, c)
                        tot_pred[..., t] = u_pred

                pred_test_mse += (
                    batch_mse_rel_fn(tot_pred[..., start:], images[..., start:])
                    .mean()
                    .item()
                    * n_samples
                )
                mse += (
                    (tot_pred[..., start:] - images[..., start:]) ** 2
                ).mean().item() * n_samples

                if seq_extra_len > 0:
                    mse_in += (
                        (
                            tot_pred[..., start:seq_inter_len]
                            - images[..., start:seq_inter_len]
                        )
                        ** 2
                    ).mean().item() * n_samples
                    mse_out += (
                        (tot_pred[..., -seq_extra_len:] - images[..., -seq_extra_len:])
                        ** 2
                    ).mean().item() * n_samples

            code_test_mse = code_test_mse / (ntest * t_dim)
            pred_test_mse = pred_test_mse / ntest
            mse = mse / ntest
            mse_in = mse_in / ntest
            mse_out = mse_out / ntest

            wandb.log(
                {
                    "code_test_mse": code_test_mse,
                    "code_train_mse": code_train_mse,
                    "pred_train_mse": pred_train_mse,
                    "pred_test_mse": pred_test_mse,
                    "mse_trajectory": mse,
                    "mse_in": mse_in,
                    "mse_out": mse_out,
                }
            )
        else:
            wandb.log(
                {
                    "code_train_mse": code_train_mse,
                },
                step=step,
                commit=not step_show,
            )

        if True in (step_show, step_show_last):
            if code_train_mse < best_loss:
                best_loss = code_train_mse
                if T_train != T_test:
                    torch.save(
                        {
                            "cfg": cfg,
                            "epoch": step,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "grid_tr": grid_tr,
                            "grid_te": grid_te,
                        },
                        f"{model_dir}/{run_name}.pt",
                    )
                if T_train == T_test:
                    torch.save(
                        {
                            "cfg": cfg,
                            "epoch": step,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "loss": code_test_mse,
                            "grid_tr": grid_tr,
                            "grid_te": grid_te,
                        },
                        f"{model_dir}/{run_name}.pt",
                    )

    return pred_train_mse


if __name__ == "__main__":
    main()
