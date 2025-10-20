# Copyright (C) 2025 Igor Sivchek
# Licensed under the MIT License.
# See license text at [https://opensource.org/license/mit].

"""
Tools to work with Sourav Banerjee's Animal Image Dataset (90 Different
Animals) v5.

Requires Python >= 3.12.

References:
1. "Animal Image Dataset (90 Different Animals)", by Sourav Banerjee, v5,
   2022.07.17.
   https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals
2. "Datasets & DataLoaders", PyTorch v2.7.0+cu126, 2024.11.05.
   https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
3. "pathlib - Object-oriented filesystem paths", Python v3.13.3, 2025.06.02.
   https://docs.python.org/3/library/pathlib.html
4. "Built-in Exceptions", Python v3.13.3, 2025.06.02.
   https://docs.python.org/3/library/exceptions.html
"""

from copy import deepcopy
import pathlib
from typing import Any, Callable, Optional, Union

import PIL
import timm
import torch
import torchvision

__all__ = ["Animal90", "DatasetMaker", "augment_data"]


DATASET_PATH = "../data/banerjee-animal-90"
DATASET_DIR = "animals"
# LABEL_FILE = "names-of-the-animals.txt"


# Type aliases for better readability.
LabelIdx = int
ImageIdx = int
ActLabelIdx = int  # Active label index.
ActImageIdx = int  # Active image index.
Label = str
ImageName = str


# It has a method to select only necessary classes from a dataset.
# Each label has its own unique internal number. External (active) label
# numbers have sequantial numeration after selecting a subset of labels. These
# external numbers may differ from the internal numbers.
class Animal90(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        dataset_path: pathlib.Path,
        images: list[tuple[ImageName, LabelIdx]],
        labels: list[tuple[Label, ImageIdx, ImageIdx]],
        pil_images: bool = True,
        transform: Optional[Callable] = None
    ) -> None:
        self.__dataset_path = dataset_path
        # All images and labels.
        self.__labels = labels
        self.__images = images
        # Output data type of images: PIL (True) or Tensor (False).
        self.__pil_images = pil_images
        # A transformation applied to each image.
        self.__transform = transform
        # Provide images with only specified label indexes.
        self.__act_labels: list[tuple[LabelIdx, ActImageIdx, ActImageIdx]]
        self.__act_images: list[tuple[ImageIdx, ActLabelIdx]]
        self.use_labels()

    def __len__(self) -> ActImageIdx:
        return len(self.__act_images)

    def __getitem__(
        self,
        idx: ActImageIdx
    ) -> tuple[Union[torch.Tensor, PIL.ImageFile.ImageFile], ActLabelIdx]:
        # The main drawback here is that it loads an image by each call.
        image_idx, act_label_idx = self.__act_images[idx]
        image_name, label_idx = self.__images[image_idx]
        label = self.__labels[label_idx][0]
        image_path = self.__dataset_path.joinpath(label, image_name)
        if self.__pil_images:  # image: PIL.ImageFile.ImageFile
            image = PIL.Image.open(image_path)
        else:  # image: torch.Tensor
            image = torchvision.io.decode_image(image_path)
        if self.__transform is not None:
            image = self.__transform(image)
        return image, act_label_idx

    def get_num_labels(self) -> LabelIdx:
        return len(self.__labels)

    def get_num_act_labels(self) -> ActLabelIdx:
        return len(self.__act_labels)

    def act_label_idx_to_label_idx(
        self,
        act_label_idx: ActLabelIdx
    ) -> LabelIdx:
        return self.__act_labels[act_label_idx][0]

    def get_label(self, act_label_idx: ActLabelIdx) -> Label:
        label_idx = self.__act_labels[act_label_idx][0]
        return self.__labels[label_idx][0]

    def get_labels(self) -> list[Label]:
        # O(n), where n = len(self.__labels).
        return [label for label, _, _ in self.__labels]

    def get_act_labels(self) -> list[tuple[Label, LabelIdx]]:
        # O(n), where n = len(self.__labels) (the worst case).
        return [
            (self.__labels[label_idx][0], label_idx)
            for label_idx, _, _
            in self.__act_labels
        ]

    def get_image_path(self, act_image_idx: ActImageIdx) -> pathlib.Path:
        image_idx = self.__act_images[act_image_idx][0]
        image_name, label_idx = self.__images[image_idx]
        label = self.__labels[label_idx][0]
        return self.__dataset_path.joinpath(label, image_name)

    def get_num_images(self) -> ImageIdx:
        return len(self.__images)

    def get_num_images_by_label(
        self,
        act_label_idx: ActLabelIdx
    ) -> ActImageIdx:
        begin_act_image_idx, end_act_image_idx = (
            self.__act_labels[act_label_idx][1:3]
        )
        return end_act_image_idx - begin_act_image_idx

    def get_state(self) -> dict[str, list]:
        # Warning: it does not contain image transformations.
        return dict(
            images=deepcopy(self.__images),
            labels=deepcopy(self.__labels),
            act_labels=[label_idx for label_idx, _, _ in self.__act_labels],
        )

    def get_transform(self) -> Optional[Callable]:
        return self.__transform

    def set_transform(self, transform: Optional[Callable]) -> None:
        self.__transform = transform

    def use_labels(
        self,
        *,
        label_indexes: Optional[list[LabelIdx]] = None,
        share: Optional[float] = None
    ) -> None:
        # share: 0.1112 -> 10 labels out of 90; 0.89 -> 80 labels out of 90.
        if label_indexes is not None and share is not None:
            raise ValueError(
                "An attempt to select labels by indexes and by share simultaneously."
            )
        if share is not None:
            if share < 0.0 or 1.0 < share:
                raise ValueError(
                    f"A share must be in [0, 1] interval, but given: {share}"
                )
            num_act_labels = int(share * len(self.__labels))
            label_indexes = (
                torch.randperm(n=len(self.__labels))
                .narrow(dim=0, start=0, length=num_act_labels)
                .tolist()
            )
        if label_indexes is None:
            # Use all labels by default.
            label_indexes = list(range(len(self.__labels)))
        if not isinstance(label_indexes, list):
            label_indexes = list(label_indexes)
        label_indexes.sort()
        self.__act_labels = list()
        self.__act_images = list()
        for act_label_idx, label_idx in enumerate(label_indexes):
            label, begin_image_idx, end_image_idx = self.__labels[label_idx]
            begin_act_image_idx = len(self.__act_images)
            self.__act_images.extend([
                (image_idx, act_label_idx)
                for image_idx
                in range(begin_image_idx, end_image_idx)
            ])
            end_act_image_idx = len(self.__act_images)
            self.__act_labels.append((
                label_idx, begin_act_image_idx, end_act_image_idx
            ))

    def set_image_type(self, t: str) -> None:
        t = t.lower()
        if t == "pil":
            self.__pil_images = True
        elif t == "tsr" or t == "tensor":
            self.__pil_images = False
        else:
            raise ValueError(f"An unknown image type: {t}")


class DatasetMaker:
    def __init__(self, *, dataset_path: Union[str, pathlib.Path]) -> None:
        if not isinstance(dataset_path, pathlib.Path):
            dataset_path = pathlib.Path(dataset_path)
        self.__dataset_path = dataset_path
        self.__labels: list[tuple[Label, ImageIdx, ImageIdx]]
        self.__images: list[tuple[ImageName, LabelIdx]]
        # Set ``labels`` and ``images`` variables.
        self.__enum_images()

    def make_datasets(
        self,
        *,
        train_transform: Optional[Callable] = None,
        eval_transform: Optional[Callable] = None,
        valid_share: float = 0.1,
        test_share: float = 0.1
    ) -> None:
        # train_share = 1.0 - valid_share - test_share
        # Check that the share values are in [0, 1] interval.
        # Check that there is at least one image per label.
        train_images = list()  # list[tuple[ImageName, LabelIdx]]
        valid_images = list()
        test_images = list()
        train_labels = list()  # list[tuple[LabelName, ImageIdx, ImageIdx]]
        valid_labels = list()
        test_labels = list()
        for label, begin_image_idx, end_image_idx in self.__labels:
            num_images = end_image_idx - begin_image_idx
            num_valid_images = int(num_images * valid_share)
            num_test_images = int(num_images * test_share)
            num_train_images = num_images - num_valid_images - num_test_images
            indexes = torch.randperm(n=num_images)
            indexes.add_(begin_image_idx)
            train_indexes, valid_indexes, test_indexes = indexes.split([
                num_train_images, num_valid_images, num_test_images
            ])
            # train_indexes.sort()
            # valid_indexes.sort()
            # test_indexes.sort()
            # Training dataset.
            begin_train_image_idx = len(train_images)
            train_images.extend([
                self.__images[train_idx] for train_idx in train_indexes
            ])
            end_train_image_idx = len(train_images)
            train_labels.append((
                label, begin_train_image_idx, end_train_image_idx
            ))
            # Validation dataset.
            begin_valid_image_idx = len(valid_images)
            valid_images.extend([
                self.__images[valid_idx] for valid_idx in valid_indexes
            ])
            end_valid_image_idx = len(valid_images)
            valid_labels.append((
                label, begin_valid_image_idx, end_valid_image_idx
            ))
            # Test dataset.
            begin_test_image_idx = len(test_images)
            test_images.extend([
                self.__images[test_idx] for test_idx in test_indexes
            ])
            end_test_image_idx = len(test_images)
            test_labels.append((
                label, begin_test_image_idx, end_test_image_idx
            ))
        # A training dataset may have data augmentation.
        train_ds = Animal90(
            dataset_path=self.__dataset_path, images=train_images,
            labels=train_labels, transform=train_transform
        )
        # There should be no data augmentation in validation and test datasets.
        valid_ds = Animal90(
            dataset_path=self.__dataset_path, images=valid_images,
            labels=valid_labels, transform=eval_transform
        )
        test_ds = Animal90(
            dataset_path=self.__dataset_path, images=test_images,
            labels=test_labels, transform=eval_transform
        )
        return train_ds, valid_ds, test_ds

    def restore_datasets(
        self,
        *,
        train_ds_state: Optional[dict[str, list]] = None,
        valid_ds_state: Optional[dict[str, list]] = None,
        test_ds_state: Optional[dict[str, list]] = None,
        train_transform: Optional[Callable] = None,
        eval_transform: Optional[Callable] = None,
    ) -> None:
        if train_ds_state is not None:
            train_ds = Animal90(
                dataset_path=self.__dataset_path,
                images=train_ds_state["images"],
                labels=train_ds_state["labels"],
                transform=train_transform
            )
            train_ds.use_labels(label_indexes=train_ds_state["act_labels"])
        else:
            train_ds = None
        if valid_ds_state is not None:
            valid_ds = Animal90(
                dataset_path=self.__dataset_path,
                images=valid_ds_state["images"],
                labels=valid_ds_state["labels"],
                transform=eval_transform
            )
            valid_ds.use_labels(label_indexes=valid_ds_state["act_labels"])
        else:
            valid_ds = None
        if test_ds_state is not None:
            test_ds = Animal90(
                dataset_path=self.__dataset_path,
                images=test_ds_state["images"],
                labels=test_ds_state["labels"],
                transform=eval_transform
            )
            test_ds.use_labels(label_indexes=test_ds_state["act_labels"])
        else:
            test_ds = None
        return train_ds, valid_ds, test_ds

    def get_num_labels(self) -> LabelIdx:
        return len(self.__labels)

    def __enum_images(self) -> None:
        # ``pathlib.Path.walk`` added in Python 3.12.
        dataset_iter = pathlib.Path.walk(self.__dataset_path)  # ->
        # -> (dirpath, dirnames, filenames)
        # next(dataset_iter) ->
        # -> (Path("<path>/animals"), ["antelope", ..., "zebra"], [])
        labels = next(dataset_iter)[1]
        self.__labels = list()
        self.__images = list()
        for label_idx, (root, dirs, files) in enumerate(dataset_iter):
            # next(dataset_iter) ->
            # -> (Path("<path>/animals/<label>"), [], ["<image.jpg>", ...])
            # Each label: (label, begin_image_idx, end_image_idx)
            label = labels[label_idx]
            begin_image_idx = len(self.__images)
            for file in files:
                self.__images.append((file, label_idx))
            end_image_idx = len(self.__images)
            self.__labels.append((label, begin_image_idx, end_image_idx))


def augment_data(
    data_config: dict[str, Any],
) -> torchvision.transforms.transforms.Compose:
    # data_config["input_size"] -> (channels, height, width)
    cropping_size = data_config["input_size"][1:3]
    resizing_size = int(1.05 * max(cropping_size))
    data_mean = data_config["mean"]
    data_std = data_config["std"]
    return torchvision.transforms.transforms.Compose([
        torchvision.transforms.transforms.Resize(
            size=resizing_size,
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            max_size=None,
            antialias=True
        ),
        torchvision.transforms.transforms.CenterCrop(size=cropping_size),
        torchvision.transforms.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.transforms.ColorJitter(
            brightness=(0.6, 1.4),
            contrast=(0.6, 1.4),
            saturation=(0.6, 1.4)
        ),
        timm.data.transforms.MaybeToTensor(),
        torchvision.transforms.transforms.Normalize(
            mean=torch.Tensor(data_mean),
            std=torch.Tensor(data_std)
        ),
    ])


if __name__ == "__main__":
    dataset_path = pathlib.Path(DATASET_PATH) / DATASET_DIR
    dsm = DatasetMaker(dataset_path=dataset_path)
    train_ds, valid_ds, test_ds = dsm.make_datasets()
