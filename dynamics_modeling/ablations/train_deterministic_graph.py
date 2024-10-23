import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
print(sys.executable)
import einops
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
    DiagonalGaussianDistribution,
)
import torch

from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR

# from aroma.DIT import DiT
from aroma.DIT_deterministic import DiT

from diffusers.schedulers import DDPMScheduler
from torch.cuda.amp import GradScaler


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def linear_scheduler(start, end, num_steps):
    delta = (end - start) / num_steps
    return [start + i * delta for i in range(num_steps)]


class PadCollateToken:
    """
    A custom collator that pads tensors to the maximum length in the batch.
    """

    def __init__(self, pad_idx=0):
        self.pad_idx = pad_idx  # The padding value to use, e.g., 0 for numerical data

    def pad_collate(self, batch):
        """
        Pads and collates batches of tensors that have different lengths into a single tensor.
        """
        # Find the longest tensor
        images_list = [b["images"] for b in batch]
        pos_list = [b["coords"] for b in batch]
        max_len = max(pos.size(0) for pos in pos_list)
        mask = [
            torch.ones(pos.shape[0]) for pos in pos_list
        ]  # we create a mask of 1 for real values and 0 for padded
        mask = [
            torch.nn.functional.pad(x, (0, max_len - x.size(0)), value=0) for x in mask
        ]
        padded_pos = [
            torch.nn.functional.pad(
                pos, (0, 0, 0, 0, 0, max_len - pos.size(0)), value=0
            )
            for pos in pos_list
        ]
        padded_images = [
            torch.nn.functional.pad(
                images, (0, 0, 0, 0, 0, max_len - images.size(0)), value=-100
            )
            for images in images_list
        ]
        pos_tensor = torch.stack(padded_pos)
        images_tensor = torch.stack(padded_images)
        mask_tensor = torch.stack(mask)

        return {
            "latent_codes": torch.stack([b["latent_codes"] for b in batch]),
            "mean": torch.stack([b["mean"] for b in batch]),
            "logvar": torch.stack([b["logvar"] for b in batch]),
            "images": images_tensor,
            "coords": pos_tensor,
            "idx": [b["idx"] for b in batch],
            "mask": mask_tensor,
            "sequence_size": [b["sequence_size"] for b in batch],
        }

    def __call__(self, batch):
        return self.pad_collate(batch)


class TokenDataset(Dataset):
    def __init__(self, dataset, mean, logvar, size_boost=1):
        """
        Args:
            data (list): List of data items.
            labels (list): List of labels for the data items.
        """
        self.mean = mean
        self.logvar = logvar
        self.dataset = dataset
        self.size_boost = size_boost

    def __len__(self):
        return len(self.mean) * self.size_boost

    def __getitem__(self, idx):
        graph, idx = self.dataset.__getitem__(idx)

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

        return {
            "latent_codes": codes,
            "mean": mu,
            "logvar": logvar,
            "images": graph.images,
            "coords": graph.pos,
            "idx": idx,
            "sequence_size": sequence_size,
        }


@hydra.main(config_path="config/", config_name="transformer_cylinder.yaml")
def main(cfg: DictConfig) -> None:

    # neceassary for some reason now
    torch.set_default_dtype(torch.float32)  # try amp later

    # data
    dataset_name = cfg.data.dataset_name
    ntrain = cfg.data.ntrain
    ntest = cfg.data.ntest
    seed = cfg.data.seed

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

    # dynamics
    # push_forward = cfg.dynamics.push_forward

    print("run dir given", run_dir)

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
    os.makedirs(model_dir, exist_ok=True)

    set_seed(seed)

    if cfg.data.dataset_name == "cylinder-flow":
        trainset = CylinderFlowDataset(
            path="/data/serrano/meshgraphnet/cylinder_flow",
            split="train",
            noise=0.02,
            task="sliced",
            mode="full",
        )

        valset = CylinderFlowDataset(
            path="/data/serrano/meshgraphnet/cylinder_flow",
            split="valid",
            noise=0.02,
            task="sliced",
            mode="full",
        )

    elif cfg.data.dataset_name == "airfoil-flow":
        trainset = AirfoilFlowDataset(
            path="/data/serrano/meshgraphnet/airfoil",
            split="train",
            noise=0.02,
            task="sliced",
            mode="full",
        )

        valset = AirfoilFlowDataset(
            path="/data/serrano/meshgraphnet/airfoil",
            split="valid",
            noise=0.02,
            task="sliced",
            mode="full",
        )
    else:
        raise NotImplementedError

    # total frames trainset
    ntrain = len(trainset)
    # total frames testset
    ntest = len(valset)

    pad_collate = PadCollate()
    batch_size_tmp = 1

    # create torch dataset
    train_loader_tmp = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size_tmp,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=pad_collate,
    )

    test_loader_tmp = torch.utils.data.DataLoader(
        valset,
        batch_size=batch_size_tmp,
        shuffle=False,
        num_workers=1,
        collate_fn=pad_collate,
    )

    T_train = trainset.T
    T_test = valset.T

    checkpoint = torch.load(cfg.wandb.checkpoint_path)
    cfg_inr = checkpoint["cfg"]

    num_latents = cfg_inr.inr.num_latents
    latent_dim = cfg_inr.inr.latent_dim  # latent_dim usually

    mu_train = torch.zeros(ntrain, num_latents, latent_dim, T_train)
    logvar_train = torch.zeros(ntrain, num_latents, latent_dim, T_train)

    mu_test = torch.zeros(ntest, num_latents, latent_dim, T_test)
    logvar_test = torch.zeros(ntest, num_latents, latent_dim, T_test)

    num_channels = 3 if cfg.data.dataset_name == "cylinder-flow" else 4
    input_dim = 2

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

    for substep, (coords, images, mask, idx) in enumerate(train_loader_tmp):
        images = images.cuda()
        coords = coords.cuda()
        mask = mask.cuda()
        n_samples = images.shape[0]
        T = images.shape[-1]

        images = einops.rearrange(images, "b ... c t -> (t b) (...) c")
        coords = einops.rearrange(coords, "b ... c t -> (t b) (...) c")

        with torch.no_grad():
            features, logvar = inr.encode(images, coords)

        mean = einops.rearrange(features, "(t b) n c -> b n c t", t=T)
        # logvar = einops.rearrange(logvar, '(t b) n c -> b n c t', t=T)

        mu_train[idx] = mean.cpu().detach()
        # logvar_train[idx] = logvar.cpu().detach()
    print("done encoding train")

    for substep, (coords, images, mask, idx) in enumerate(test_loader_tmp):
        images = images.cuda()
        coords = coords.cuda()
        mask = mask.cuda()
        n_samples = images.shape[0]
        T = images.shape[-1]

        images = einops.rearrange(images, "b ... c t -> (t b) (...) c")
        coords = einops.rearrange(coords, "b ... c t -> (t b) (...) c")

        with torch.no_grad():
            features, logvar = inr.encode(images, coords)

        mean = einops.rearrange(features, "(t b) n c -> b n c t", t=T)
        # logvar = einops.rearrange(logvar, '(t b) n c -> b n c t', t=T)

        mu_test[idx] = mean.cpu().detach()
        # logvar_train_extra[idx] = logvar.cpu().detach()

    pad_collate_tkn = PadCollateToken()

    train_dataset_tkn = TokenDataset(trainset, mu_train, logvar_train)
    test_dataset_tkn = TokenDataset(valset, mu_test, logvar_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset_tkn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=pad_collate_tkn,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset_tkn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=pad_collate_tkn,
    )

    model = DiT(
        input_size=latent_dim,
        num_tokens=num_latents,
        in_channels=4,
        hidden_size=128,  # 192,#1152,
        depth=4,  # 4
        num_heads=4,  # 6
        mlp_ratio=4.0,  # 4.0
        learn_sigma=False,
    ).cuda()

    print("num params", count_parameters(model))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )

    scheduler_lr = StepLR(optimizer, step_size=epochs // 5, gamma=0.5)
    # Initialize the gradient scaler

    difference_weight = 1
    predict_difference = False
    use_statistics = True

    best_loss = np.inf

    print("use statistics", use_statistics)

    for step in range(epochs):
        step_show = step % (epochs // 20) == 0
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

                if use_statistics:
                    x = mean_batch[..., t]
                else:
                    x = codes[..., t]

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
                pred = model(
                    torch.cat([x, y_noised], axis=1), torch.zeros((x.shape[0],)).to(x)
                )
                target = y
                # print('pred', pred.shape, target.shape, x_in.shape)
                loss = (
                    (pred - target) ** 2
                ).mean()  # self.train_criterion(pred, target

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                code_train_mse += loss.item() * n_samples

        code_train_mse = code_train_mse / (ntrain * t_dim)

        if True in (step_show, step_show_last):
            pred_train_mse = pred_train_mse / (ntrain * t_dim)

        scheduler_lr.step()

        if True in (step_show, step_show_last):
            pred_test_mse = 0
            code_test_mse = 0

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

                    for t in range(t_dim - 1 - start):
                        y_noised = torch.zeros_like(x).to(
                            x
                        )  # , dtype=x.dtype, device=x.device
                        pred = model(
                            torch.cat([x, y_noised], axis=1),
                            torch.zeros((x.shape[0],)).to(x),
                        )
                        z_pred[..., t + 1] = pred.detach()
                        x = pred.detach()
                        if predict_difference:
                            y = y * difference_weight + x

                        code_test_mse += (
                            (mean_batch[..., start + t + 1] - pred) ** 2
                        ).mean().item() * n_samples

                pred = z_pred.detach()  # [..., :t_dim]
                images = batch["images"].cuda()  # [..., :t_dim]
                coords = batch["coords"].cuda()  # [..., :t_dim]
                mask = batch["mask"].cuda()
                images = images.reshape(n_samples, -1, num_channels, t_dim)
                coords = coords.reshape(n_samples, -1, input_dim, t_dim)  # * 2 - 1
                tot_pred = torch.zeros_like(images)

                for t in range(start, t_dim):
                    x = pred[..., t]
                    c = coords[..., t]
                    y = images[..., t]

                    with torch.no_grad():
                        # print('x', x.shape, c.shape)
                        u_pred = inr.decode(x, c)
                        tot_pred[..., t] = u_pred

                # print('tot_pred', tot_pred.shape, images.shape, mask.shape)
                pred_test_mse += (
                    (
                        ((tot_pred[..., start:t_dim] - images[..., start:t_dim]) ** 2)
                        * mask[..., None, None]
                    ).sum()
                    / (mask[..., None, None].sum() * num_channels * t_dim)
                    * n_samples
                )

            code_test_mse = code_test_mse / (ntest * t_dim)
            pred_test_mse = pred_test_mse / ntest

            wandb.log(
                {
                    "code_test_mse": code_test_mse,
                    "code_train_mse": code_train_mse,
                    "pred_train_mse": pred_train_mse,
                    "pred_test_mse": pred_test_mse,
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

        if code_test_mse < best_loss:
            best_loss = code_test_mse
            if T_train != T_test:
                torch.save(
                    {
                        "cfg": cfg,
                        "epoch": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
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
                    },
                    f"{model_dir}/{run_name}.pt",
                )

    return pred_train_mse


if __name__ == "__main__":
    main()
