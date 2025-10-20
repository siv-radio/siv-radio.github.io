# Copyright (C) 2025 Igor Sivchek
# Licensed under the MIT License.
# See license text at [https://opensource.org/license/mit].

"""
Utilities for storing (saving and loading) embeddings and checkpoints.
"""

import pathlib
from typing import Any, Callable, Optional, Union

import numpy
import pandas
import timm
import torch

import banerjee

__all__ = [
    "save_embeddings", "load_embeddings", "save_checkpoint", "load_checkpoint"
]


def save_embeddings(
    *,
    embeddings: numpy.ndarray,
    targes: numpy.ndarray,
    filename: Union[str, pathlib.Path],
    inform: bool = False
) -> None:
    filename = pathlib.Path(filename)
    embeddings_df = pandas.DataFrame(data=embeddings)
    embeddings_df.insert(loc=0, column='target', value=targes)
    embeddings_df.to_csv(
        f"{filename}.zip",
        index=False,
        compression={
            'method': 'zip',
            'archive_name': f"{filename.name}.csv",
        }
    )
    if inform:
        print(
            "Embeddings have been saved:",
            f"{filename}.zip",
            sep="\n  ",
        )


def load_embeddings(
    *,
    filename: Union[str, pathlib.Path],
    inform: bool = False
) -> tuple[numpy.ndarray]:
    filename = pathlib.Path(filename)
    embeddings_df = pandas.read_csv(filename)
    targets = embeddings_df.iloc[:, 0].to_numpy()
    embeddings = embeddings_df.iloc[:, 1:].to_numpy()
    if inform:
        print(
            "Embeddings have been loaded:",
            filename,
            sep="\n  ",
        )
    return targets, embeddings


def save_checkpoint(
    *,
    filename: Union[str, pathlib.Path],
    model: torch.nn.Module,
    train_ds: Optional[banerjee.Animal90] = None,
    valid_ds: Optional[banerjee.Animal90] = None,
    test_ds: Optional[banerjee.Animal90] = None,
    inform=False,
) -> None:
    # Warning: it does not save image trainsformations of the datasets.
    state = dict(
        model_config = model.default_cfg,  # May not be universal.
        model=model.state_dict(),
        train_ds_state=(train_ds.get_state() if train_ds is not None else None),
        valid_ds_state=(valid_ds.get_state() if valid_ds is not None else None),
        test_ds_state=(test_ds.get_state() if test_ds is not None else None),
    )
    torch.save(obj=state, f=filename)
    if inform:
        print(
            "A ckeckpoint has been saved:",
            f"{filename}",
            sep="\n  ",
        )


def load_checkpoint(
    *,
    checkpoint_path: Union[str, pathlib.Path],
    dataset_path: Optional[Union[str, pathlib.Path]],
    inform=False,
) -> tuple:
    state = torch.load(
        f=checkpoint_path, map_location="cpu", weights_only=True
    )
    model_state = state["model"]
    if "head.linear.weight" in model_state:  # EfficientViT.
        num_labels = len(model_state["head.linear.weight"])
    elif "classifier.weight" in model_state:  # MobileNetV4.
        num_labels = len(model_state["classifier.weight"])
    elif "head.fc.weight" in model_state:  # TinyViT.
        num_labels = len(model_state["head.fc.weight"])
    else:
        if inform:
            print("state['model'] keys:", model_state.keys())
        raise ValueError("An unexpected model. Cannot define 'num_classes' attribute.")
    model = timm.create_model(
        model_name=state["model_config"]["hf_hub_id"],
        pretrained=False,  # Do not load model parameters.
        num_classes=num_labels,  # Set the model head.
    )
    model.load_state_dict(model_state)
    train_ds_state = state["train_ds_state"]
    valid_ds_state = state["valid_ds_state"]
    test_ds_state = state["test_ds_state"]
    if (
        dataset_path is not None and
        (
            train_ds_state is not None or
            valid_ds_state is not None or
            test_ds_state is not None
        )
    ):
        dsm = banerjee.DatasetMaker(dataset_path=dataset_path)
        train_ds, valid_ds, test_ds = dsm.restore_datasets(
            train_ds_state=train_ds_state,
            valid_ds_state=valid_ds_state,
            test_ds_state=test_ds_state,
        )
    else:
        train_ds = valid_ds = test_ds = None
    if inform:
        print(
            "A ckeckpoint has been loaded:",
            f"{checkpoint_path}",
            sep="\n  ",
        )
    return model, train_ds, valid_ds, test_ds
