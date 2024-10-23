import os
import random
import shelve
from pathlib import Path

import einops
import h5py
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F

# import xarray as xr
from scipy import io
from torch.utils.data import Dataset

# MP-PDE imports
#from equations.PDEs import CE
#from common.utils import HDF5Dataset
from aroma.utils.data.setting import init_setting
from typing import Tuple

class PDE:
    """Generic PDE template"""
    def __init__(self):
        # Data params for grid and initial conditions
        super().__init__()
        pass

    def __repr__(self):
        return "PDE"

    def FDM_reconstruction(self, t: float, u: torch.Tensor) -> torch.Tensor:
        """A finite differences method template"""
        pass

    def FVM_reconstruction(self, t: float, u: torch.Tensor) -> torch.Tensor:
        """A finite volumes method template"""
        pass

    def WENO_reconstruction(self, t: float, u: torch.Tensor) -> torch.Tensor:
        """A WENO reconstruction template"""


class HDF5Dataset(Dataset):
    """Load samples of an PDE Dataset, get items according to PDE"""

    def __init__(self,
                 path: str,
                 pde: PDE,
                 mode: str,
                 base_resolution: list=None,
                 super_resolution: list=None,
                 load_all: bool=False) -> None:
        """Initialize the dataset object
        Args:
            path: path to dataset
            pde: string of PDE ('CE' or 'WE')
            mode: [train, valid, test]
            base_resolution: base resolution of the dataset [nt, nx]
            super_resolution: super resolution of the dataset [nt, nx]
            load_all: load all the data into memory
        Returns:
            None
        """
        super().__init__()
        f = h5py.File(path, 'r')
        self.mode = mode
        self.pde = pde
        self.dtype = torch.float64
        self.data = f[self.mode]
        self.base_resolution = (250, 100) if base_resolution is None else base_resolution
        self.super_resolution = (250, 200) if super_resolution is None else super_resolution
        self.dataset_base = f'pde_{self.base_resolution[0]}-{self.base_resolution[1]}'
        self.dataset_super = f'pde_{self.super_resolution[0]}-{self.super_resolution[1]}'

        ratio_nt = self.data[self.dataset_super].shape[1] / self.data[self.dataset_base].shape[1]
        ratio_nx = self.data[self.dataset_super].shape[2] / self.data[self.dataset_base].shape[2]
        assert (ratio_nt.is_integer())
        assert (ratio_nx.is_integer())
        self.ratio_nt = int(ratio_nt)
        self.ratio_nx = int(ratio_nx)

        self.nt = self.data[self.dataset_base].attrs['nt']
        self.dt = self.data[self.dataset_base].attrs['dt']
        self.dx = self.data[self.dataset_base].attrs['dx']
        self.x = self.data[self.dataset_base].attrs['x']
        self.tmin = self.data[self.dataset_base].attrs['tmin']
        self.tmax = self.data[self.dataset_base].attrs['tmax']

        if load_all:
            data = {self.dataset_super: self.data[self.dataset_super][:]}
            f.close()
            self.data = data


    def __len__(self):
        return self.data[self.dataset_super].shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Get data item
        Args:
            idx (int): data index
        Returns:
            torch.Tensor: numerical baseline trajectory
            torch.Tensor: downprojected high-resolution trajectory (used for training)
            torch.Tensor: spatial coordinates
            list: equation specific parameters
        """
        if(f'{self.pde}' == 'CE'):
            # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
            u_super = self.data[self.dataset_super][idx][::self.ratio_nt][None, None, ...]
            left = u_super[..., -3:-1]
            right = u_super[..., 1:3]
            u_super_padded = torch.tensor(np.concatenate((left, u_super, right), -1))
            weights = torch.tensor([[[[0.2]*5]]])
            u_super = F.conv1d(u_super_padded, weights, stride=(1, self.ratio_nx)).squeeze().numpy()
            x = self.x

            # Base resolution trajectories (numerical baseline) and equation specific parameters
            u_base = self.data[self.dataset_base][idx]
            variables = {}
            variables['alpha'] = self.data['alpha'][idx]
            variables['beta'] = self.data['beta'][idx]
            variables['gamma'] = self.data['gamma'][idx]

            return u_base, u_super, x, variables

        elif(f'{self.pde}' == 'WE'):
            # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
            # No padding is possible due to non-periodic boundary conditions
            weights = torch.tensor([[[[1./self.ratio_nx]*self.ratio_nx]]])
            u_super = self.data[self.dataset_super][idx][::self.ratio_nt][None, None, ...]
            u_super = F.conv1d(torch.tensor(u_super), weights, stride=(1, self.ratio_nx)).squeeze().numpy()

            # To match the downprojected trajectories, also coordinates need to be downprojected
            x_super = torch.tensor(self.data[self.dataset_super].attrs['x'][None, None, None, :])
            x = F.conv1d(x_super, weights, stride=(1, self.ratio_nx)).squeeze().numpy()

            # Base resolution trajectories (numerical baseline) and equation specific parameters
            u_base = self.data[self.dataset_base][idx]
            variables = {}
            variables['bc_left'] = self.data['bc_left'][idx]
            variables['bc_right'] = self.data['bc_right'][idx]
            variables['c'] = self.data['c'][idx]

            return u_base, u_super, x, variables

        else:
            raise Exception("Wrong experiment")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dynamics_data(
    data_dir,
    dataset_name,
    ntrain,
    ntest,
    seq_inter_len=20,
    seq_extra_len=20,
    sub_from=1,
    sub_tr=1,
    sub_te=1,
    same_grid=True,
):
    """Get training and test data as well as associated coordinates, depending on the dataset name.

    Args:
        data_dir (str): path to the dataset directory
        dataset_name (str): dataset name (e.g. "navier-stokes)
        ntrain (int): number of training samples
        ntest (int): number of test samples
        sub_tr (int or float, optional): when set to int > 1, subsamples x as x[::sub_tr]. When set to float < 1, subsamples x as x[index] where index is a random index of shape int(sub_tr*len(x)). Defaults to 1.
        sub_tr (int or float, optional): when set to int > 1, subsamples x as x[::sub_te]. When set to float < 1, subsamples x as x[index] where index is a random index of shape int(sub_te*len(x)). Defaults to 1.
        same_grid (bool, optional): If True, all the trajectories avec the same grids.

    Raises:
        NotImplementedError: _description_

    Returns:
        u_train (torch.Tensor): (ntrain, ..., T)
        u_test (torch.Tensor): (ntest, ..., T)
        grid_tr (torch.Tensor): coordinates of u_train
        grid_te (torch.Tensor): coordinates of u_test
    """

    data_dir = Path(data_dir)

    u_train_out = None
    u_test_out = None
    u_train_ext = None
    u_test_ext = None
    grid_tr_out = None
    grid_te_out = None
    grid_tr_ext = None
    grid_te_ext = None

    if dataset_name == "navier-stokes-1e-3":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_fno(
            data_dir / "ns_V1e-3_N5000_T50.mat",
            ntrain,
            ntest,
            seq_inter_len,
            seq_extra_len,
            50,
        )
        u_train = u_train[..., 9:]
        u_eval_extrapolation = u_eval_extrapolation[..., 9:]
        u_test = u_test[..., 9:]

    elif dataset_name == "navier-stokes-1e-4":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_fno(
            data_dir / "ns_V1e-4_N10000_T30.mat",
            ntrain,
            ntest,
            seq_inter_len,
            seq_extra_len,
            30,
        )
        u_train = u_train[..., 9:]
        u_eval_extrapolation = u_eval_extrapolation[..., 9:]
        u_test = u_test[..., 9:]

    elif dataset_name == "navier-stokes-1e-5":
        # index_start = 9
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_fno(
            data_dir / "NavierStokes_V1e-5_N1200_T20.mat",
            1000,
            200,
            seq_inter_len,
            seq_extra_len,
        )
        u_train = u_train[..., 9:]
        u_eval_extrapolation = u_eval_extrapolation[..., 9:]
        u_test = u_test[..., 9:]

    elif dataset_name == "navier-stokes-dino":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_dino(
            data_dir, seq_inter_len, seq_extra_len
        )

    elif dataset_name == "shallow-water-dino":
        u_train, u_eval_extrapolation, u_test = get_shallow_water_dino(
            data_dir, seq_inter_len, seq_extra_len
        )

    elif dataset_name == "mp-pde-burgers":
        u_train, u_eval_extrapolation, u_test = get_mp_pde_burgers(
            data_dir, seq_inter_len, seq_extra_len, ntrain=ntrain, ntest=ntest
        )
    else:
        raise NotImplementedError

    # u_train should be of shape (N, ..., C, T)
    if dataset_name in ["shallow-water-dino"]:
        grid_tr = shape2spherical_coordinates(u_train.shape[1:-2])
        grid_tr_extra = shape2spherical_coordinates(u_eval_extrapolation.shape[1:-2])
        grid_te = shape2spherical_coordinates(u_test.shape[1:-2])

    else:
        grid_tr = shape2coordinates(u_train.shape[1:-2])
        grid_tr_extra = shape2coordinates(u_eval_extrapolation.shape[1:-2])
        grid_te = shape2coordinates(u_test.shape[1:-2])
    if u_train_out is not None:
        grid_tr_out = shape2coordinates(u_train_out.shape[1:-2])
        grid_te_out = shape2coordinates(u_test_out.shape[1:-2])
    if u_train_ext is not None:
        grid_tr_ext = shape2coordinates(u_train_ext.shape[1:-2])
        grid_te_ext = shape2coordinates(u_test_ext.shape[1:-2])

    # we need to artificially create a time dimension for the coordinates

    grid_tr = einops.repeat(
        grid_tr, "... -> b ... t", t=u_train.shape[-1], b=u_train.shape[0]
    )
    grid_tr_extra = einops.repeat(
        grid_tr_extra,
        "... -> b ... t",
        t=u_eval_extrapolation.shape[-1],
        b=u_eval_extrapolation.shape[0],
    )
    grid_te = einops.repeat(
        grid_te, "... -> b ... t", t=u_test.shape[-1], b=u_test.shape[0]
    )
    if isinstance(sub_from, int):
        grid_tr = dynamics_subsample(grid_tr, sub_from)
        u_train = dynamics_subsample(u_train, sub_from)

    if isinstance(sub_from, int):
        grid_tr_extra = dynamics_subsample(grid_tr_extra, sub_from)
        u_eval_extrapolation = dynamics_subsample(u_eval_extrapolation, sub_from)

    if isinstance(sub_from, int):
        grid_te = dynamics_subsample(grid_te, sub_from)
        u_test = dynamics_subsample(u_test, sub_from)

    if isinstance(sub_from, int):
        grid_tr = dynamics_subsample(grid_tr, sub_from)
        u_train = dynamics_subsample(u_train, sub_from)

    if isinstance(sub_tr, int):
        grid_tr = dynamics_subsample(grid_tr, sub_tr)
        u_train = dynamics_subsample(u_train, sub_tr)

    if isinstance(sub_tr, int):
        grid_tr_extra = dynamics_subsample(grid_tr_extra, sub_tr)
        u_eval_extrapolation = dynamics_subsample(u_eval_extrapolation, sub_tr)

    if isinstance(sub_te, int):
        grid_te = dynamics_subsample(grid_te, sub_te)
        u_test = dynamics_subsample(u_test, sub_te)

    if isinstance(sub_tr, float) and (sub_tr < 1):
        if same_grid:
            tmp = einops.rearrange(u_train, "b ... c t -> b (...) c t")
            num_points = tmp.shape[1]
            perm = torch.randperm(num_points)
            mask_tr = perm[: int(sub_tr * len(perm))].clone().sort()[0]
            grid_tr = dynamics_subsample(grid_tr, mask_tr)
            u_train = dynamics_subsample(u_train, mask_tr)

        else:
            print("computing different grids")
            u_train, grid_tr, perm = dynamics_different_subsample(
                u_train, grid_tr, sub_tr
            )

    if isinstance(sub_tr, float) and (sub_tr < 1):
        if same_grid:
            tmp = einops.rearrange(u_eval_extrapolation, "b ... c t -> b (...) c t")
            num_points = tmp.shape[1]
            perm = torch.randperm(num_points)
            mask_tr_eval = perm[: int(sub_tr * len(perm))].clone().sort()[0]
            grid_tr_extra = dynamics_subsample(grid_tr_extra, mask_tr_eval)
            u_eval_extrapolation = dynamics_subsample(
                u_eval_extrapolation, mask_tr_eval
            )

        else:
            u_eval_extrapolation, grid_tr_extra, perm = dynamics_different_subsample(
                u_eval_extrapolation, grid_tr_extra, sub_tr
            )

    if isinstance(sub_te, float) and (sub_te < 1):
        if same_grid:
            tmp = einops.rearrange(u_test, "b ... c t -> b (...) c t")
            num_points = tmp.shape[1]
            perm = torch.randperm(num_points)
            mask_te = perm[: int(sub_te * len(perm))].clone().sort()[0]
            grid_te = dynamics_subsample(grid_te, mask_te)
            u_test = dynamics_subsample(u_test, mask_te)

        else:
            u_test, grid_te, perm = dynamics_different_subsample(
                u_test, grid_te, sub_te
            )

    return u_train, u_eval_extrapolation, u_test, grid_tr, grid_tr_extra, grid_te


def get_shallow_water(filename, ntrain, ntest, min_sub=1):
    path_to_file = os.path.join(filename, "../shallow_water/data_t0180_freq6_N1200.h5")
    rf = h5py.File(path_to_file, "r")

    # shape (N, T, long, lat)
    # shape (1200, 6, 256, 128)

    initial_time = 0
    target_time = 5

    height_scale = 3 * 1e3
    vorticity_scale = 2

    height = torch.Tensor(rf["height"][()])
    vorticity = torch.Tensor(rf["vorticity"][()])

    # permute long and lat
    # create an extra dimension
    height = (height_scale * height).permute(0, 1, 3, 2).unsqueeze(-1)
    vorticity_scale = (vorticity_scale * vorticity).permute(0, 1, 3, 2).unsqueeze(-1)

    x_train = torch.cat(
        [height[:ntrain, initial_time], vorticity_scale[:ntrain, initial_time]], axis=-1
    )
    y_train = torch.cat(
        [height[:ntrain, target_time], vorticity_scale[:ntrain, target_time]], axis=-1
    )
    x_test = torch.cat(
        [height[-ntest:, initial_time], vorticity_scale[-ntest:, initial_time]], axis=-1
    )
    y_test = torch.cat(
        [height[-ntest:, target_time], vorticity_scale[-ntest:, target_time]], axis=-1
    )

    return x_train, y_train, x_test, y_test


def get_navier_stokes_fno(
    filename, ntrain=1000, ntest=200, seq_inter_len=10, seq_extra_len=10, total_seq=None
):
    # reader = MatReader(data_dir / "NavierStokes_V1e-5_N1200_T20.mat")
    reader = MatReader(filename)
    u = reader.read_field("u")
    print("u", u.shape)

    # u of shape (N, Dx, Dy, T)
    u_train = u[:ntrain]
    u_test = u[-ntest:]
    if total_seq is not None:
        u_train = u_train[..., :total_seq]
        u_test = u_test[..., :total_seq]

    T = u_train.shape[-1]

    u_train = u_train.unsqueeze(-2)
    u_test = u_test.unsqueeze(-2)
    # seq_inter_len = seq_inter_len - 9
    # seq_extra_len = seq_extra_len - 9

    return split_data(u_train, u_test, seq_inter_len, seq_extra_len, T)


def get_navier_stokes_dino(filename, seq_inter_len=20, seq_extra_len=20):
    train_path = str(filename) + "/dino/navier_1e-3_256_2_train.shelve"
    test_path = str(filename) + "/dino/navier_1e-3_256_2_test.shelve"

    data_train = dict(shelve.open(str(train_path)))
    data_test = dict(shelve.open(str(test_path)))

    # data_train.pop("a")
    # data_train.pop("t")
    # data_test.pop("a")
    # data_test.pop("t")

    # concatenate dictionaries to be of shape (ntrain, 40, 256, 256)
    #  u = einops.rearrange(u, 'b (t d) w l -> (b d) t w l', d=2)
    u_train = torch.tensor(
        np.concatenate(
            list(
                map(
                    lambda key: np.array(data_train[key]["data"]),
                    data_train.keys(),
                )
            )
        )
    )
    u_test = torch.tensor(
        np.concatenate(
            list(
                map(
                    lambda key: np.array(data_test[key]["data"]),
                    data_test.keys(),
                )
            )
        )
    )

    u_train = einops.rearrange(u_train, "b t w h -> b w h t").unsqueeze(-2)
    u_test = einops.rearrange(u_test, "b t w h -> b w h t").unsqueeze(-2)

    # output of shape (N, Dx, Dy, 1, T)

    return split_data(u_train, u_test, seq_inter_len, seq_extra_len, 40)


def split_data(u_train, u_test, seq_inter_len, seq_extra_len, total_seq):
    if (seq_inter_len is not None) & (seq_extra_len is not None):
        if total_seq % (seq_inter_len + seq_extra_len) == 0:
            u_train = einops.rearrange(
                u_train, "b ... (d t) -> (b d) ... t", t=seq_inter_len + seq_extra_len
            )
            u_test = einops.rearrange(
                u_test, "b ... (d t) -> (b d) ... t", t=seq_inter_len + seq_extra_len
            )
        elif total_seq % (seq_inter_len + seq_extra_len) != 0:
            u_train = u_train[
                ...,
                total_seq
                - total_seq
                // (seq_inter_len + seq_extra_len)
                * (seq_inter_len + seq_extra_len) :,
            ]
            u_test = u_test[
                ...,
                total_seq
                - total_seq
                // (seq_inter_len + seq_extra_len)
                * (seq_inter_len + seq_extra_len) :,
            ]
            u_train = einops.rearrange(
                u_train, "b ... (d t) -> (b d) ... t", t=seq_inter_len + seq_extra_len
            )
            u_test = einops.rearrange(
                u_test, "b ... (d t) -> (b d) ... t", t=seq_inter_len + seq_extra_len
            )
    u_eval_extrapolation = u_train
    u_train = u_train[..., :seq_inter_len]
    return u_train, u_eval_extrapolation, u_test


def get_shallow_water_dino(filename, seq_inter_len=20, seq_extra_len=20):
    train_path = str(filename) + "/dino/shallow_water_16_160_128_256_train.h5"
    test_path = str(filename) + "/dino/shallow_water_2_160_128_256_test.h5"

    with h5py.File(train_path, "r") as f:
        vorticity_train = f["vorticity"][()]
        height_train = f["height"][()]

    with h5py.File(test_path, "r") as f:
        vorticity_test = f["vorticity"][()]
        height_test = f["height"][()]

    # shape (N, T, long, lat)
    # train shape (16, 160, 256, 128)
    # test shape (2, 160, 256, 128)

    height_scale = 3 * 1e3
    vorticity_scale = 2

    vorticity_train = torch.from_numpy(vorticity_train).float() * vorticity_scale
    vorticity_test = torch.from_numpy(vorticity_test).float() * vorticity_scale

    height_train = torch.from_numpy(height_train).float() * height_scale
    height_test = torch.from_numpy(height_test).float() * height_scale

    u_train = torch.cat(
        [height_train.unsqueeze(-1), vorticity_train.unsqueeze(-1)], axis=-1
    )
    u_test = torch.cat(
        [height_test.unsqueeze(-1), vorticity_test.unsqueeze(-1)], axis=-1
    )

    u_train = einops.rearrange(u_train, "b t long lat c -> b lat long c t")
    u_test = einops.rearrange(u_test, "b t long lat c -> b lat long c t")

    return split_data(u_train, u_test, seq_inter_len, seq_extra_len, 160)


class MatReader(object):
    """Loader for navier-stokes data"""

    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except BaseException:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x


def set_seed(seed=33):
    """Set all seeds for the experiments.

    Args:
        seed (int, optional): seed for pseudo-random generated numbers.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def dynamics_subsample(x, sub=1, missing_batch=False):
    """
    WARNING: This functions does not work for graph data.

    Subsample data and coordinates in the same way.

    Args:
        x (torch.Tensor): data to be subsampled, of shape (N, Dx, Dy, C, T)
        sub (int or Tensor, optional): When set to int, subsamples x as x[::sub]. When set to Tensor of indices, slices x in the 1st dim. Defaults to 1.
        missing_batch (bool, optional): Coordinates are missing batch dimension at this stage and should be aligned with data wehn set to True. Defaults to True.

    Returns:
        x (torch.Tensor): subsampled array.
    """

    if missing_batch:
        x = x.unsqueeze(0)
    if isinstance(sub, int):
        # regular slicing
        if x.ndim == 4:  # 1D data (N, Dx, C, T)
            x = x[:, ::sub]
        if x.ndim == 5:  # 2D data (N, Dx, Dy, C, T)
            x = x[:, ::sub, ::sub]

    if isinstance(sub, torch.Tensor):
        x = einops.rearrange(
            x, "b ... c t -> b (...) c t"
        )  # x.reshape(x.shape[0], -1, x.shape[-1])
        x = x[:, sub]

    if missing_batch:
        x = x.squeeze(0)
    return x


def dynamics_different_subsample(u, grid, draw_ratio):
    """
    Performs subsampling for univariate time series
    Args:
        u (torch.Tensor): univariate time series (batch_size, num_points, num_channels, T)
        grid (torch.Tensor): timesteps coordinates (batch_size, num_points, input_dim)
        draw_ratio (float): draw ratio
    Returns:
        small_data: subsampled data
        small_grid: subsampled grid
        permutations: draw indexs
    """
    u = einops.rearrange(u, "b ... c t -> b (...) c t")
    grid = einops.rearrange(grid, "b ... c t -> b (...) c t")

    N = u.shape[0]
    C = u.shape[-2]
    dims = grid.shape[-2]
    T = u.shape[-1]
    input_dim = grid.shape[-2]
    partial_grid_size = int(draw_ratio * grid.shape[1])

    # Create draw indexes
    permutations = [
        torch.randperm(grid.shape[1])[:partial_grid_size].unsqueeze(0)
        for ii in range(N)
    ]
    permutations = torch.cat(permutations, axis=0)
    small_u = torch.gather(
        u, 1, permutations.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, C, T)
    )
    small_grid = torch.gather(
        grid, 1, permutations.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, dims, T)
    )

    return small_u, small_grid, permutations


def shape2coordinates(spatial_shape, max_value=1.0):
    """Create coordinates from a spatial shape.

    Args:
        spatial_shape (list): Shape of data, i.e. [64, 64] for navier-stokes.

    Returns:
        grid (torch.Tensor): Coordinates that span (0, 1) in each dimension.
    """
    coords = []
    for i in range(len(spatial_shape)):
        coords.append(torch.linspace(0.0, max_value, spatial_shape[i]))
    # Tensor will have shape (*spatial_shape, len(spatial_shape))
    return torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)


def shape2circular_coordinates(spatial_shape):
    """Create coordinates from a spatial shape.

    Args:
        spatial_shape (list): Shape of data, i.e. [64, 64] for navier-stokes.

    Returns:
        grid (torch.Tensor): Coordinates that span (0, 1) in each dimension.
    """
    coords = []
    for i in range(len(spatial_shape)):
        coords.append(torch.linspace(0.0, 2 * np.pi, spatial_shape[i]))
    # Tensor will have shape (*spatial_shape, len(spatial_shape))
    coords = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)

    new_coords = torch.zeros(*coords.shape[:-1], 2)
    new_coords[..., 0] = torch.sin(coords[..., 0])
    new_coords[..., 1] = torch.cos(coords[..., 0])
    return new_coords


def shape2spherical_coordinates(spatial_shape):
    """Returns spherical coordinates on a uniform latitude and longitude grid.
    Args:
        spatial_shape (tuple of int): Tuple (num_lats, num_lons) containing
            number of latitudes and longitudes in grid.
    """
    num_lats, num_lons = spatial_shape
    # Uniformly spaced latitudes and longitudes corresponding to ERA5 grids
    latitude = torch.linspace(90.0, -90.0, num_lats)
    longitude = torch.linspace(0.0, 360.0 - (360.0 / num_lons), num_lons)
    # Create a grid of latitude and longitude values (num_lats, num_lons)
    longitude_grid, latitude_grid = torch.meshgrid(longitude, latitude, indexing="xy")
    # Create coordinate tensor
    # Spherical coordinates have 3 dimensions
    coordinates = torch.zeros(latitude_grid.shape + (3,))
    long_rad = deg_to_rad(longitude_grid)
    lat_rad = deg_to_rad(latitude_grid)
    coordinates[..., 0] = torch.cos(lat_rad) * torch.cos(long_rad)
    coordinates[..., 1] = torch.cos(lat_rad) * torch.sin(long_rad)
    coordinates[..., 2] = torch.sin(lat_rad)
    return coordinates


def deg_to_rad(degrees):
    return torch.pi * degrees / 180.0


def rad_to_deg(radians):
    return 180.0 * radians / torch.pi


def repeat_coordinates(coordinates, batch_size):
    """Repeats the coordinate tensor to create a batch of coordinates.
    Args:
        coordinates (torch.Tensor): Shape (*spatial_shape, len(spatial_shape)).
        batch_size (int): If not zero, repeats the coordinate tensor to create
            a batch of coordinates.
    """
    if batch_size:
        ones_like_shape = (1,) * coordinates.ndim
        return coordinates.unsqueeze(0).repeat(batch_size, *ones_like_shape)
    else:
        return coordinates


def get_mp_pde_burgers(
    data_dir,
    seq_inter_len=30,
    seq_extra_len=20,
    ntrain=2048,
    ntest=128,
    subsample_time_every=1,
    starting_timestep=None,
    visualize_data=False,
    experiment="E1",
    base_resolution=(250, 200),
):

    #pde = CE(device=device)
    pde = "burgers"
    train_string = f"{data_dir}/{pde}_train_{experiment}.h5"
    valid_string = f"{data_dir}/{pde}_valid_{experiment}.h5"
    test_string = f"{data_dir}/{pde}_test_{experiment}.h5"

    super_resolution = base_resolution

    train_dataset = HDF5Dataset(
        train_string,
        pde=pde,
        mode="train",
        base_resolution=base_resolution,
        super_resolution=super_resolution,
        load_all=True,
    )
    valid_dataset = HDF5Dataset(
        valid_string,
        pde=pde,
        mode="valid",
        base_resolution=base_resolution,
        super_resolution=super_resolution,
        load_all=True,
    )
    test_dataset = HDF5Dataset(
        test_string,
        pde=pde,
        mode="test",
        base_resolution=base_resolution,
        super_resolution=super_resolution,
        load_all=True,
    )

    def get_smoothed_data_from_dataset(dataset, subsample_time_every):
        u_super = dataset.data[dataset.dataset_super][:][:: dataset.ratio_nt][
            None, None, ...
        ]
        print(u_super.shape)  # (1, 1, 128, 250, 200)
        left = u_super[..., -3:-1]
        print(left.shape)  # (1, 1, 128, 250, 2)
        right = u_super[..., 1:3]
        print(right.shape)  # (1, 1, 128, 250, 2)
        u_super_padded = torch.tensor(np.concatenate((left, u_super, right), -1))
        u_super_padded = u_super_padded.float()
        print(
            f"u_super_padded: {u_super_padded.shape}"
        )  # torch.Size([1, 1, 128, 250, 204])
        u_super_padded = u_super_padded.squeeze(0)
        print(
            f"u_super_padded: {u_super_padded.shape}"
        )  # torch.Size([1, 128, 250, 204])
        u_super_padded = u_super_padded.permute((1, 0, 2, 3))
        print(
            f"u_super_padded: {u_super_padded.shape}"
        )  # torch.Size([128, 1, 250, 204])
        # weights = torch.tensor([[[[0.2]*5]]], dtype=torch.float32)
        weights = torch.tensor([[[0.2] * 5]], dtype=torch.float32).repeat(250, 1, 1)
        # pdb.set_trace()
        print(
            f"u_super: {u_super_padded.shape}",
            u_super_padded.mean(),
            u_super_padded.std(),
        )
        # u_super = F.conv1d(u_super_padded, weights, stride=(1, dataset.ratio_nx)).squeeze().numpy()
        u_super = (
            F.conv1d(
                u_super_padded.squeeze(), weights, stride=dataset.ratio_nx, groups=250
            )
            .squeeze()
            .numpy()
        )
        print(
            f"u_super: {u_super.shape}", u_super.mean(), u_super.std()
        )  # (batch, time, space) = (128, 250, 200)
        # u_super is now the same shape (although unbatched) as in MP-PDE
        # Still need to permute the axes to match the other datasets
        u_super = einops.rearrange(u_super, "b t x -> b x t")
        print(f"u_super : {u_super.shape}")
        if starting_timestep is not None:
            u_super = u_super[..., starting_timestep:]
        u_super = u_super[..., ::subsample_time_every]
        print(f"u_super : {u_super.shape}")
        # if sequence_length is not None:
        #    u_super = einops.rearrange(u_super, 'b x (d t) -> (b d) x t', t=sequence_length)
        # u_super = einops.rearrange(u_super, 'b x t -> b x 1 1 t') # b x 1 1 t before
        print(f"u_super : (batch, space, 1, 1, time)")
        print(f"u_super : {u_super.shape}")
        return u_super

    u_train = get_smoothed_data_from_dataset(
        train_dataset, subsample_time_every=subsample_time_every
    )
    # pdb.set_trace()
    # u_valid = get_smoothed_data_from_dataset(valid_dataset)
    u_test = get_smoothed_data_from_dataset(
        test_dataset, subsample_time_every=subsample_time_every
    )

    u_train = u_train[:ntrain] if ntrain is not None else u_train
    u_test = u_test[:ntest] if ntest is not None else u_test

    u_train = u_train[..., None, :]  # add one channel
    u_test = u_test[..., None, :]  # add one channel

    print(f"u_train : {u_train.shape}")
    print(f"u_test : {u_test.shape}")

    # Testing visualization
    if visualize_data:
        import matplotlib.pyplot as plt

        for i in np.random.randint(0, u_train.shape[0], 10):
            for t in range(u_train.shape[-1]):
                plt.plot(u_train[i, :, 0, 0, t], label=f"t={t}")
            plt.legend()
            plt.show()
            plt.savefig(f"runs/img/u_train_{i}_{t}.png", bbox_inches="tight")
            print(f"Plotted trajectories {i} for u_train")

    # output of shape (N, Dx, Dy=1, C=1, T)
    # return u_train, u_test
    return split_data(
        u_train, u_test, seq_inter_len, seq_extra_len, seq_inter_len + seq_extra_len
    )
