import functools
import json
import os
import random
import shelve
from pathlib import Path

import einops
import h5py
import numpy as np
import scipy.io
import tensorflow as tf
import torch
import torch.nn.functional as F
import xarray as xr
from scipy import io
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

KEY_TO_INDEX = {
    "cylinder-flow": {"pressure": 0, "vx": 1, "vy": 2},
    "airfoil-flow": {"pressure": 0, "density": 1, "vx": 2, "vy": 3},
}

KEY_TO_STATS = {
    "airfoil-flow": {
        "pressure": {"mean": 99297.1250, "std": 11169.3750},
        "density": {"mean": 1.2018, "std": 0.1094},
        "velocity": {
            "mean": torch.Tensor([81.3912, 78.6263]),
            "std": torch.Tensor([105.0405, 104.8035]),
        },
        "image": {
            "mean": torch.Tensor([81.3912, 78.6263, 99297.1250, 1.2018]),
            "std": torch.Tensor([105.0405, 104.8035, 11169.3750, 0.1094]),
        },
        "pos": {"min": -20, "max": 20},
    },
    "cylinder-flow": {
        "pressure": {"mean": 0.0444, "std": 0.3522},
        "velocity": {
            "mean": torch.Tensor([0.2847, 0.2825]),
            "std": torch.Tensor([0.5032, 0.5025]),
        },
        "image": {
            "mean": torch.Tensor([0.2847, 0.2825, 0.0444]),
            "std": torch.Tensor([0.5032, 0.5025, 0.3522]),
        },
        "pos": {"min": 0, "max": 1.60},
    },
}


def _parse(proto, meta):
    """Parses a trajectory from tf.Example."""
    feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in meta["field_names"]}
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta["features"].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field["dtype"]))
        data = tf.reshape(data, field["shape"])
        if field["type"] == "static":
            data = tf.tile(data, [meta["trajectory_length"], 1, 1])
        elif field["type"] == "dynamic_varlen":
            length = tf.io.decode_raw(features["length_" + key].values, tf.int32)
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
        elif field["type"] != "dynamic":
            raise ValueError("invalid data format")
        out[key] = data
    return out


def load_dataset(path, split):
    """Load dataset."""
    with open(os.path.join(path, "meta.json"), "r") as fp:
        meta = json.loads(fp.read())
    ds = tf.data.TFRecordDataset(os.path.join(path, split + ".tfrecord"))
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
    ds = ds.prefetch(1)
    return ds


class CylinderFlowDataset(Dataset):
    """Custom dataset for encoding task. Contains the values, the codes, and the coordinates for input and output."""

    def __init__(
        self,
        path="/data/serrano/meshgraphnet/cylinder_flow",
        split="train",
        noise=0.02,
        task="slice",
        mode="random_t",
    ):
        """
        Args:
            x (torch.Tensor): Dataset input values
            y (torch.Tensor): Dataset output values
            grid_x (torch.Tensor): input coordinates
            grid_y (torch.Tensor): output coordinates
            latent_dim (int, optional): Latent dimension of the code. Defaults to 256.
        """
        super().__init__(None, None, None)
        self.path = path
        self.split = split
        self.T = 40 if split == "train" else 60
        self.mode = mode
        assert mode in ["random_t", "full"]

        if task == "sliced":
            self.dataset = self.load_sliced_data()
        else:
            raise NotImplementedError

    def load_sliced_data(self):
        filename = Path(self.path) / f"sliced_{self.split}.h5"
        self._indices = []
        try:
            f = h5py.File(filename, "r")
            f.close()

        except:
            ds = load_dataset(self.path, self.split)
            f = h5py.File(filename, "w")
            # we take the first and last timestamps
            for index, d in enumerate(ds):
                # (t, n, c) -> (n, c, t)
                pos = d["mesh_pos"].numpy()[::10].transpose(1, 2, 0)
                node_type = d["node_type"].numpy()[::10].transpose(1, 2, 0)
                velocity = d["velocity"].numpy()[::10].transpose(1, 2, 0)
                cells = d["cells"].numpy()[::10].transpose(1, 2, 0)
                pressure = d["pressure"].numpy()[::10].transpose(1, 2, 0)

                data = ("pos", "node_type", "velocity", "cells", "pressure")
                # d = f.create_dataset(str(index), (len(data), ), dtype=pos.dtype)
                g = f.create_group(str(index))
                for k in data:
                    g[k] = eval(k)
            f.close()

        dataset = {}
        with h5py.File(filename, "r") as f:
            for index in range(len(f)):
                self._indices.append(index)
                v = torch.from_numpy(f[f"{index}"]["velocity"][()])
                p = torch.from_numpy(f[f"{index}"]["pressure"][()])
                pos = torch.from_numpy(f[f"{index}"]["pos"][()])
                node_type = torch.from_numpy(f[f"{index}"]["node_type"][()])
                cells = torch.from_numpy(f[f"{index}"]["cells"][()])
                tmp = Data(pos=pos)
                tmp.v = v
                tmp.p = p
                tmp.node_type = node_type
                tmp.cells = cells
                dataset[f"{index}"] = tmp

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        graph = self.dataset[f"{key}"].clone()
        graph.images = torch.cat([graph.v, graph.p], axis=1).float()
        graph = self.normalize(graph)
        if self.mode == "random_t":
            index_t = np.random.randint(0, self.T)
            graph.pos = graph.pos[..., index_t].unsqueeze(-1)
            graph.images = graph.images[..., index_t].unsqueeze(-1)

        elif self.mode == "full":
            index_t = self.T
            graph.pos = graph.pos[..., :index_t]
            graph.images = graph.images[..., :index_t]

        graph.pos = graph.pos.float()

        return (graph, key)
        # return (graph[..., np.random.randint(0, self.T)].unsqueeze(-1), key) if self.mode == "random_t" else (graph, key)

    def normalize(self, graph):
        tmp_dic = KEY_TO_STATS["cylinder-flow"]
        mu, sigma = tmp_dic["image"]["mean"], tmp_dic["image"]["std"]
        pos_max = tmp_dic["pos"]["max"]

        graph.images = (graph.images - mu[None, ..., None]) / sigma[None, ..., None]
        graph.pos = (graph.pos / pos_max - 0.5) * 2  # rescale in -1, 1

        return graph


class AirfoilFlowDataset(Dataset):
    """Custom dataset for encoding task. Contains the values, the codes, and the coordinates for input and output."""

    def __init__(
        self,
        path="/data/serrano/meshgraphnet/airfoil",
        split="train",
        noise=0.02,
        task="sliced",
        mode="random_t",
    ):
        """
        Args:
            x (torch.Tensor): Dataset input values
            y (torch.Tensor): Dataset output values
            grid_x (torch.Tensor): input coordinates
            grid_y (torch.Tensor): output coordinates
            latent_dim (int, optional): Latent dimension of the code. Defaults to 256.
        """
        super().__init__(None, None, None)
        self.path = path
        self.split = split
        self.T = 40 if split == "train" else 61
        self.mode = mode
        assert mode in ["random_t", "full"]

        if task == "sliced":
            self.dataset = self.load_sliced_data()
        else:
            raise NotImplementedError

    def load_sliced_data(self):
        filename = Path(self.path) / f"sliced_{self.split}.h5"
        self._indices = []
        try:
            f = h5py.File(filename, "r")
            f.close()

        except:
            ds = load_dataset(self.path, self.split)
            f = h5py.File(filename, "w")
            # we take the first and last timestamps
            for index, d in enumerate(ds):
                # (t, n, c) -> (n, c, t)
                pos = d["mesh_pos"].numpy()[::10].transpose(1, 2, 0)
                node_type = d["node_type"].numpy()[::10].transpose(1, 2, 0)
                velocity = d["velocity"].numpy()[::10].transpose(1, 2, 0)
                cells = d["cells"].numpy()[::10].transpose(1, 2, 0)
                density = d["density"].numpy()[::10].transpose(1, 2, 0)
                pressure = d["pressure"].numpy()[::10].transpose(1, 2, 0)
                data = ("pos", "node_type", "velocity", "cells", "density", "pressure")
                # d = f.create_dataset(str(index), (len(data), ), dtype=pos.dtype)
                g = f.create_group(str(index))
                for k in data:
                    g[k] = eval(k)
            f.close()

        dataset = {}
        with h5py.File(filename, "r") as f:
            for index in range(len(f)):
                self._indices.append(index)
                v = torch.from_numpy(f[f"{index}"]["velocity"][()])
                p = torch.from_numpy(f[f"{index}"]["pressure"][()])
                rho = torch.from_numpy(f[f"{index}"]["density"][()])
                pos = torch.from_numpy(f[f"{index}"]["pos"][()])
                node_type = torch.from_numpy(f[f"{index}"]["node_type"][()])
                cells = torch.from_numpy(f[f"{index}"]["cells"][()])
                tmp = Data(pos=pos)
                tmp.v = v
                tmp.p = p
                tmp.rho = rho
                tmp.node_type = node_type
                tmp.cells = cells
                dataset[f"{index}"] = tmp

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        graph = self.dataset[f"{key}"].clone()
        graph.images = torch.cat(
            [
                graph.v,
                graph.p,
                graph.rho,
            ],
            axis=1,
        ).float()
        graph = self.normalize(graph)
        if self.mode == "random_t":
            index_t = np.random.randint(0, self.T)
            graph.pos = graph.pos[..., index_t].unsqueeze(-1)
            graph.images = graph.images[..., index_t].unsqueeze(-1)
        elif self.mode == "full":
            index_t = self.T
            graph.pos = graph.pos[..., :index_t]
            graph.images = graph.images[..., :index_t]

        graph.pos = graph.pos.float()

        return (graph, key)

    def normalize(self, graph):
        tmp_dic = KEY_TO_STATS["airfoil-flow"]
        mu, sigma = tmp_dic["image"]["mean"], tmp_dic["image"]["std"]
        pos_max = tmp_dic["pos"]["max"]

        graph.images = (graph.images - mu[None, ..., None]) / sigma[None, ..., None]
        graph.pos = graph.pos / pos_max

        return graph


class PadCollate:
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
        # index = torch.Tensor([key for _, key in batch])
        max_len = max(graph.pos.size(0) for graph, _ in batch)
        #        for j in range(len(batch)):
        # Pad each tensor to the max length and stack them
        mask = [
            torch.ones(graph.pos.shape[0]) for graph, _ in batch
        ]  # we create a mask of 1 for real values and 0 for padded
        mask = [
            torch.nn.functional.pad(x, (0, max_len - x.size(0)), value=0) for x in mask
        ]
        padded_pos = [
            torch.nn.functional.pad(
                graph.pos, (0, 0, 0, 0, 0, max_len - graph.pos.size(0)), value=0
            )
            for graph, _ in batch
        ]
        padded_images = [
            torch.nn.functional.pad(
                graph.images,
                (0, 0, 0, 0, 0, max_len - graph.images.size(0)),
                value=-100,
            )
            for graph, _ in batch
        ]
        # print("toto", [x.shape for x, _, _ in batch])
        # print("tata", [x.shape for x in padded_x])
        pos_tensor = torch.stack(padded_pos)
        images_tensor = torch.stack(padded_images)
        mask_tensor = torch.stack(mask)

        # y_tensor = torch.tensor([y for _, y in batch])

        return (
            pos_tensor.float(),
            images_tensor.float(),
            mask_tensor.float(),
            [key for _, key in batch],
        )

    def __call__(self, batch):
        return self.pad_collate(batch)
