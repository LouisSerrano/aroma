from pathlib import Path
import einops
import torch
from torch.utils.data import Dataset

KEY_TO_INDEX = {
    "shallow-water-dino": {"height": 0, "vorticity": 1},
    "navier-stokes-dino": {"vorticity": 0},
    "navier-stokes-1e-5": {"vorticity": 0},
    "navier-stokes-1e-4": {"vorticity": 0},
    "mp-pde-burgers": {"vorticity": 0},
}
def rearrange(set, dataset_name):
    if dataset_name == "shallow-water-dino":
        set.v = einops.rearrange(set.v, "N ... T -> (N T) ... 1")
    else:
        set.v = einops.rearrange(set.v, "N ... T -> (N T) ...")
    set.c = einops.rearrange(set.c, "N ... T -> (N T) ... ")
    return set

class TemporalDataset(Dataset):
    """Custom dataset for encoding task. Contains the values, the codes, and the coordinates."""

    def __init__(
        self,
        v,
        grid,
        dataset_name=None,
        subsample=0,
    ):
        """
        Args:
            v (torch.Tensor): Dataset values, with shape (N Dx Dy C T). Where N is the
            number of trajectories, Dx the size of the first spatial dimension, Dy the size
            of the second spatial dimension, C the number of channels (ususally 1), and T the
            number of timestamps.
            grid (torch.Tensor): Coordinates, with shape (N Dx Dy 2). We suppose that we have
            same grid over time.
            latent_dim (int, optional): Latent dimension of the code. Defaults to 64.
        """
        N = v.shape[0]
        T = v.shape[-1]
        self.v = v
        self.c = grid  # repeat_coordinates(grid, N).clone()
        self.output_dim = self.v.shape[-2]
        self.input_dim = self.c.shape[-2]
        self.T = T
        self.dataset_name = dataset_name
        self.subsample = subsample

    def __len__(self):
        return len(self.v)

    def __getitem__(self, idx):
        """The tempral dataset returns whole trajectories, identified by the index.

        Args:
            idx (int): idx of the trajectory

        Returns:
            sample_v (torch.Tensor): the trajectory with shape (Dx Dy C T)
            sample_z (torch.Tensor): the codes with shape (L T)
            sample_c (torch.Tensor): the spatial coordinates (Dx Dy 2)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_v = self.v[idx, ...]
        sample_c = self.c[idx, ...]

        return sample_v, sample_c, idx
