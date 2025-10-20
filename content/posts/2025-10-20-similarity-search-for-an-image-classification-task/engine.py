# Copyright (C) 2025 Igor Sivchek
# Licensed under the MIT License.
# See license text at [https://opensource.org/license/mit].

"""
Tools to work with artificial neural networks and datasets.
Creation and usage of a vector database.
"""

import datetime
from types import SimpleNamespace
from typing import Any, Callable, Optional, Union

import numpy as np
from sentence_transformers.util import semantic_search
import torch
from tqdm import tqdm

__all__ = ["Encoder"]


class Encoder:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        device: torch.device,
        train_dl: torch.utils.data.Dataset,
        valid_dl: torch.utils.data.Dataset,
        test_dl: torch.utils.data.Dataset,
        config: SimpleNamespace,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl
        self.config = config
        self.embeddings: Optional[torch.Tensor] = None
        self.labels: Optional[np.ndarray] = None

    # It uses all labels of a training dataset.
    def encode(self, *, inform: bool = False) -> dict[str, Any]:
        starting_time = datetime.datetime.now()

        if inform:
            print(
                "[Encode/TrainDS] Embedding creation started at: {0}"
                    .format(starting_time)
            )

        embeddings = list()
        labels = list()
        total_num_proc_batches = 0
        total_num_proc_samples = 0
        pbar = tqdm(
            iterable=range(self.train_dl.dataset.get_num_labels()),
            desc="[Encode/TrainDS]",
            leave=True,
            disable=(not inform),
            position=0,
        )
        self.model.eval()
        with torch.inference_mode():
            for label_idx in pbar:
                # Process the dataset by active labels.
                self.train_dl.dataset.use_labels(label_indexes=[label_idx])
                label_embeddings = list()
                for batch in self.train_dl:
                    # Process images with a specific label.
                    if (
                        self.config.short_run_batches is not None and
                        self.config.short_run_batches <= total_num_proc_batches
                    ):
                        if inform:
                            pbar.write(
                                "[Encode/TrainDS] A short run completed: {0} batches processed."
                                    .format(total_num_proc_batches)
                            )
                        break
                    # type(batch) -> list
                    # type(batch[0]) -> torch.Tensor
                    # batch[0].shape -> torch.Size([batch_size, num_channels, img_height, img_width])
                    # batch[0].dtype -> torch.float32
                    # type(batch[1]) -> torch.Tensor
                    # batch[1].shape -> torch.Size([batch_size])
                    # batch[1].dtype -> torch.int64
                    # Check that there is no another active label index (all
                    # active indexes are 0).
                    assert(batch[1].sum() == 0)
                    inputs = batch[0].to(self.device)
                    outputs = self.model(inputs)
                    label_embeddings.append(outputs.cpu().numpy())
                    # type(outputs) -> torch.Tensor
                    # outputs.shape -> torch.Size([batch_size, embedding_dim])
                    # outputs.dtype -> torch.float32
                    total_num_proc_batches += 1
                    total_num_proc_samples += len(batch[1])
                if len(label_embeddings) > 0:
                    label_embeddings = np.concatenate(label_embeddings)
                    average_label_embedding = label_embeddings.mean(axis=0)
                    embeddings.append(average_label_embedding)
                    labels.append(self.train_dl.dataset.get_label(0))
                else:  # Short run; last batch was not processed.
                    break
                pbar.set_description("[Encode/TrainDS] Progress")
            # Ideally, a context manager should be used here to restore the
            # original set of labels.
            self.train_dl.dataset.use_labels()  # Use all labels.
            assert(
                self.train_dl.dataset.get_num_act_labels() ==
                self.train_dl.dataset.get_num_labels()
            )

        self.labels = np.array(labels)
        self.embeddings = (
            torch
                .from_numpy(np.array(embeddings))
                .to(self.device)
        )
        assert(self.labels.shape[0] == self.embeddings.shape[0])

        num_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        embedding_dim = self.model.head_hidden_size

        ending_time = datetime.datetime.now()
        duration = ending_time - starting_time

        if inform:
            print(
                "[Encode/TrainDS] Embedding creation finished at: {0}"
                    .format(ending_time)
            )

        stats = dict(
            model_name=self.model.default_cfg["hf_hub_id"],
            num_trainable_params=num_trainable_params,
            embedding_dim=embedding_dim,
            dataset_part="train",
            num_labels=self.labels.shape[0],
            num_images=total_num_proc_samples,
            duration=duration,
        )

        if inform:
            total_num_labels = self.train_dl.dataset.get_num_labels()
            total_num_images = self.train_dl.dataset.get_num_images()
            print("-- Stats --")
            print("Model name:", stats["model_name"])
            print(
                "Number of trainable model parameters:",
                stats["num_trainable_params"]
            )
            print("Size of an embedding vector:", stats["embedding_dim"])
            print("Dataset part:", stats["dataset_part"])
            print(
                "Number of processed labels: {0}/{1}"
                    .format(stats["num_labels"], total_num_labels)
            )
            print(
                "Number of processed images: {0}/{1}"
                    .format(stats["num_images"], total_num_images)
            )
            print("Calculation duration:", stats["duration"])
            print("-----------")

        return stats

    def evaluate(
        self,
        *,
        dataset_part: str,
        inform: bool = False,
    ) -> dict[str, Any]:
        dataset_part = dataset_part.lower()
        if dataset_part == "train":
            dataloader = self.train_dl
        elif dataset_part == "valid":
            dataloader = self.valid_dl
        elif dataset_part == "test":
            dataloader = self.test_dl
        else:
            raise ValueError("An unknown dataset part: {0}".format(dataset_part))

        starting_time = datetime.datetime.now()

        dataset_part_disp = dataset_part.capitalize() + "DS"

        if inform:
            print(
                "[Eval/{0}] Accuracy calculation started at: {1}"
                    .format(dataset_part_disp, starting_time)
            )

        all_predictions = list()
        all_targets = list()
        total_num_proc_samples = 0
        total_num_corr_preds = 0
        total_num_corr_topk_preds = 0
        pbar = tqdm(
            iterable=dataloader,
            desc=f"[Eval/{dataset_part_disp}]",
            leave=True,
            disable=(not inform),
            position=0,
        )
        self.model.eval()
        with torch.inference_mode():
            for batch_idx, batch in enumerate(pbar):
                # Process the dataset by image batches.
                if (
                    self.config.short_run_batches is not None and
                    self.config.short_run_batches <= batch_idx
                ):
                    if inform:
                        pbar.write(
                            "[Eval/{0}] A short run completed: {1} batches processed."
                                .format(dataset_part_disp, batch_idx)
                        )
                    break

                inputs = batch[0].to(self.device)
                targets = batch[1]
                outputs = self.model(inputs)
                hits = semantic_search(outputs, self.embeddings, top_k=self.config.topk)

                num_corr_preds = 0
                num_corr_topk_preds = 0
                for prediction, target in zip(hits, targets):
                    predicted_label_indexes = [p["corpus_id"] for p in prediction]
                    all_predictions.append(predicted_label_indexes[0])
                    all_targets.append(target.item())
                    num_corr_preds += int(predicted_label_indexes[0] == target)
                    num_corr_topk_preds += int(target in predicted_label_indexes)

                total_num_proc_samples += len(targets)
                total_num_corr_preds += num_corr_preds
                total_num_corr_topk_preds += num_corr_topk_preds
                accuracy = total_num_corr_preds / total_num_proc_samples

                pbar.set_description(f"[Eval/{dataset_part_disp}] Accuracy: {accuracy:.4}")

        accuracy = total_num_corr_preds / total_num_proc_samples
        topk_accuracy = total_num_corr_topk_preds / total_num_proc_samples

        num_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        embedding_dim = self.model.head_hidden_size

        ending_time = datetime.datetime.now()
        duration = ending_time - starting_time

        if inform:
            print(
                "[Eval/{0}] Accuracy calculation finished at: {1}"
                    .format(dataset_part_disp, ending_time)
            )

        stats = dict(
            model_name=self.model.default_cfg["hf_hub_id"],
            num_trainable_params=num_trainable_params,
            embedding_dim=embedding_dim,
            dataset_part=dataset_part,
            targets=all_targets,
            predictions=all_predictions,
            num_labels=len(set(all_targets)),
            num_images=total_num_proc_samples,
            accuracy=accuracy,
            topk=self.config.topk,
            topk_accuracy=topk_accuracy,
            duration=duration,
        )

        if inform:
            total_num_labels = dataloader.dataset.get_num_act_labels()
            total_num_images = len(dataloader.dataset)
            print("-- Stats --")
            print("Model name:", stats["model_name"])
            print(
                "Number of trainable model parameters:",
                stats["num_trainable_params"]
            )
            print("Size of an embedding vector:", stats["embedding_dim"])
            print("Dataset part:", stats["dataset_part"])
            print(
                "Number of processed labels: {0}/{1}"
                    .format(stats["num_labels"], total_num_labels)
            )
            print(
                "Number of processed images: {0}/{1}"
                    .format(stats["num_images"], total_num_images)
            )
            print("Accuracy:", stats["accuracy"])
            print(f"Top {stats['topk']} accuracy: {stats['topk_accuracy']}")
            print("Calculation duration:", stats["duration"])
            print("-----------")

        return stats

    def infer(
        self,
        *,
        inputs: torch.Tensor,
    ) -> list[list[dict[str, Any]]]:
        # type(batch) -> list
        # type(batch[0]) -> torch.Tensor
        # batch[0].shape -> torch.Size([batch_size, num_channels, img_height, img_width])
        # batch[0].dtype -> torch.float32
        # type(batch[1]) -> torch.Tensor
        # batch[1].shape -> torch.Size([batch_size])
        # batch[1].dtype -> torch.int64
        self.model.eval()
        with torch.inference_mode():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            hits = semantic_search(outputs, self.embeddings, top_k=self.config.topk)

        return hits
