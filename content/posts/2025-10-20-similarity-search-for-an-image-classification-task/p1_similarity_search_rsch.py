# Copyright (C) 2025 Igor Sivchek
# Licensed under the MIT License.
# See license text at [https://opensource.org/license/mit].

"""
Part 1: similarity search using a pretrained model.

References:
1. "Getting Started with PyTorch Image Model (timm): a practitioner's guide",
   by Chris Hughes, 2022.02.01,
   https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055-2/
2. "Getting Started With Embeddings", by Omar Espejel, 2022.06.23.
   https://huggingface.co/blog/getting-started-with-embeddings
3. "TinyViT: Fast Pretraining Distillation for Small Vision Transformer", by
   by Wu et al., 2022.07.21.
   https://arxiv.org/abs/2207.10666
   A TinyViT image classification model. Pretrained on ImageNet-22k with
   distillation and fine-tuned on ImageNet-1k by paper authors.
   https://huggingface.co/timm/tiny_vit_21m_224.dist_in22k_ft_in1k
4. "MobileNetV4: Universal Models for the Mobile Ecosystem", by Qin et al.,
   2024.09.29.
   https://arxiv.org/abs/2404.10518
   A MobileNet-V4 image classification model. Trained on ImageNet-1k by Ross
   Wightman.
   https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k
5. "Searching for MobileNetV3", by Howard et al., 2019.11.20.
   https://arxiv.org/abs/1905.02244
   A MobileNet-v3 image classification model. Trained on ImageNet-1k in timm
   using recipe template described below.
   https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k
6. "EfficientViT: Memory Efficient Vision Transformer with Cascaded Group
   Attention", by Liu et al., 2023.05.11.
   https://arxiv.org/abs/2305.07027
   An EfficientViT (MSRA) image classification model. Trained on ImageNet-1k by
   paper authors.
   https://huggingface.co/timm/efficientvit_m0.r224_in1k
   https://huggingface.co/timm/efficientvit_m1.r224_in1k
   https://huggingface.co/timm/efficientvit_m2.r224_in1k
   https://huggingface.co/timm/efficientvit_m3.r224_in1k
   https://huggingface.co/timm/efficientvit_m4.r224_in1k
"""

import datetime
import pathlib
import random
from types import SimpleNamespace
from typing import Any, Callable, Optional, Union

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
import timm
import torch
from tqdm import tqdm

import banerjee
import cumacc
import engine
import randomness
import store
import utils
import visual


AUGMENT_ENCODE_DATA = False
AUGMENT_EVAL_DATA = False
BATCH_SIZE = 32
DATASET_PATH = "../data/banerjee-animal-90"
DATASET_DIR = "animals"
DATA_AUGMENTATION = False
EMBEDDINGS_FILENAME = "embeddings, tiny_vit_11m_224"  # 196_000 bytes
FORCE_CPU = False
# MODEL_NAME = "efficientvit_m0.r224_in1k"  # Params: 2.3M. Valid acc.: 0.67, n = 1.
# MODEL_NAME = "efficientvit_m1.r224_in1k"  # Params: 3.0M. Valid acc.: 0.82, n = 1.
# MODEL_NAME = "efficientvit_m2.r224_in1k"  # Params: 4.2M. Valid acc.: 0.80, n = 1.
# MODEL_NAME = "efficientvit_m3.r224_in1k"  # Params: 6.9M. Valid acc.: 0.69, n = 1.
# MODEL_NAME = "efficientvit_m4.r224_in1k"  # Params: 8.8M. Valid acc.: 0.68, n = 1.
# MODEL_NAME = "mobilenetv3_small_100.lamb_in1k"  # Params: 2.5M. Valid acc.: 0.84, n = 1.
# MODEL_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"  # Params: 3.8M. Valid acc.: 0.87, n = 1.
# MODEL_NAME = "tiny_vit_5m_224.dist_in22k_ft_in1k"  # Params: 5.4M. Valid acc.: 0.9513 +/- 0.0021 with 0.95 confidence, n = 20.
MODEL_NAME = "tiny_vit_11m_224.dist_in22k_ft_in1k"  # Params: 11.0M. Valid acc.: 0.9664 +/- 0.0021 with 0.95 confidence, n = 20.
# MODEL_NAME = "tiny_vit_21m_224.dist_in22k_ft_in1k"  # Params: 21.2M. Valid acc.: 0.973 +/- 0.0023 with 0.95 confidence, n = 20.
NUM_WORKERS = 0  # 0 - no multiprocessing.
PROFILE = False  # Do profiling.
PROFILING_FILENAME = "profiling-dump"
RANDOM_SEED = 17
REPRODUCIBLE = True
SAVE_EMBEDDINGS = False
SHORT_RUN = False
SHORT_RUN_BATCHES = 4
STUDENT_N = 20
# STUDENT_T = 3.182  # n = 4, 2-sided confidence 0.95.
# STUDENT_T = 2.365  # n = 8, 2-sided confidence 0.95.
STUDENT_T = 2.093  # n = 20, 2-sided confidence 0.95.
STUDENT_TWO_SIDED_CONFIDENCE = 0.95
TOPK = 4


#%% Script.
if __name__ == "__main__":
    print("Part 1: similarity search using a pretrained model.")


    #%% Create a model and a dataset maker.
    device = utils.get_device(force_cpu=FORCE_CPU)

    if REPRODUCIBLE:
        randomness.set_determinism(
            seed=RANDOM_SEED,
            use_deterministic_algorithms=False
        )
        tgen = randomness.make_torch_generator(seed=RANDOM_SEED)
    else:
        tgen = randomness.make_torch_generator()

    # https://huggingface.co/docs/timm/reference/models#timm.create_model
    model = timm.create_model(
        model_name=MODEL_NAME,
        pretrained=True,  # Load model parameters.
        num_classes=0,  # Remove the model head.
        # global_pool="",  # Remove pooling.
    )
    # "discrepancy between number of features" by Elsospi, 2024.09.13.
    # https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k/discussions/3
    # model.conv_head = torch.nn.Identity()
    # if hasattr(model, "conv_norm"):
    #     model.conv_norm = torch.nn.Identity()
    # model.to(device)
    # model.eval()
    # model.training -> False

    # Get model specific transforms (normalization, resizing).
    data_config = timm.data.resolve_model_data_config(model)
    if AUGMENT_ENCODE_DATA:
        encode_transform = banerjee.augment_data(data_config)
    else:
        encode_transform = timm.data.create_transform(
            **data_config,
            is_training=False,
        )

    if AUGMENT_EVAL_DATA:
        eval_transform = banerjee.augment_data(data_config)
    else:
        eval_transform = timm.data.create_transform(
            **data_config,
            is_training=False,
        )
    # transforms = timm.data.create_transform(**data_config, is_training=False)
    # type(transforms) -> torchvision.transforms.transforms.Compose
    # print(transforms) ->
    # Compose(
    #     Resize(size=235, interpolation=bicubic, max_size=None, antialias=True)
    #     CenterCrop(size=(224, 224))
    #     MaybeToTensor()
    #     Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
    # )
    # Note: this pipeline of transfroms is aimed to work with a PIL image
    # object as input and produces a PyTorch tensor object as output.

    dataset_path = pathlib.Path(DATASET_PATH) / DATASET_DIR

    dsm = banerjee.DatasetMaker(dataset_path=dataset_path)

    config = dict(
        short_run_batches=(SHORT_RUN_BATCHES if SHORT_RUN else None),
        topk=TOPK,
    )
    config = SimpleNamespace(**config)

    encoder = engine.Encoder(
        model=model,
        device=device,
        train_dl=None,
        valid_dl=None,
        test_dl=None,
        config=config,
    )


    #%% Create embeddings and calculate accuracy on training and validation
    # datasets.
    with utils.Profile(active=PROFILE) as profiler:
        run_results = list()
        pbar = tqdm(
            iterable=range(STUDENT_N),
            desc="[Run]",
            leave=True,
            disable=(not True),
            position=0,
        )
        for run_idx in pbar:
            train_ds, valid_ds, _ = dsm.make_datasets(
                train_transform=encode_transform,
                eval_transform=eval_transform,
                valid_share=0.2, test_share=0.0,
            )
            train_dl = torch.utils.data.DataLoader(
                dataset=train_ds, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS,
                worker_init_fn=randomness.seed_worker, generator=tgen
            )
            valid_dl = torch.utils.data.DataLoader(
                dataset=valid_ds, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS,
                worker_init_fn=randomness.seed_worker, generator=tgen
            )
            encoder.train_dl = train_dl
            encoder.valid_dl = valid_dl
            encres = encoder.encode(inform=True)
            evres_train = encoder.evaluate(dataset_part="Train", inform=True)
            evres_valid = encoder.evaluate(dataset_part="Valid", inform=True)
            run_results.append(dict(
                encode=encres, eval_train=evres_train, eval_valid=evres_valid
            ))
        profiler.dump(filename=PROFILING_FILENAME, sort_by="cumtime")


    #%% Calculate the final stats in case of a series of simulations.
    if len(run_results) > 1:
        total_num_valid_labels = valid_dl.dataset.get_num_act_labels()
        total_num_valid_images = len(valid_dl.dataset)
        total_valid_accuracy = 0.0
        total_valid_topk_accuracy = 0.0
        total_valid_duration = datetime.timedelta(0.0)
        for runres in run_results:
            total_valid_accuracy += runres["eval_valid"]["accuracy"]
            total_valid_topk_accuracy += runres["eval_valid"]["topk_accuracy"]
            total_valid_duration += runres["eval_valid"]["duration"]
        total_valid_accuracy /= len(run_results)
        total_valid_topk_accuracy /= len(run_results)
        # Standard deviations of a sample.
        total_valid_accuracy_std_dev = 0.0
        total_valid_topk_accuracy_std_dev = 0.0
        for runres in run_results:
            total_valid_accuracy_std_dev += (runres["eval_valid"]["accuracy"] - total_valid_accuracy)**2
            total_valid_topk_accuracy_std_dev += (runres["eval_valid"]["topk_accuracy"] - total_valid_topk_accuracy)**2
        # With Bessel's correction.
        total_valid_accuracy_std_dev = np.sqrt(total_valid_accuracy_std_dev / (len(run_results) - 1))
        total_valid_topk_accuracy_std_dev = np.sqrt(total_valid_topk_accuracy_std_dev / (len(run_results) - 1))
        # Confidence interval.
        total_valid_accuracy_ci = STUDENT_T * total_valid_accuracy_std_dev / np.sqrt(len(run_results))
        total_valid_topk_accuracy_ci = STUDENT_T * total_valid_topk_accuracy_std_dev / np.sqrt(len(run_results))
        final_stats = dict(
            model_name=run_results[0]["eval_valid"]["model_name"],
            num_trainable_params=run_results[0]["eval_valid"]["num_trainable_params"],
            embedding_dim=run_results[0]["eval_valid"]["embedding_dim"],
            dataset_part=run_results[0]["eval_valid"]["dataset_part"],
            num_labels=run_results[0]["eval_valid"]["num_labels"],
            num_images=run_results[0]["eval_valid"]["num_images"],
            accuracy=total_valid_accuracy,
            accuracy_confidence=STUDENT_TWO_SIDED_CONFIDENCE,
            accuracy_ci=total_valid_accuracy_ci,
            topk=run_results[0]["eval_valid"]["topk"],
            topk_accuracy=total_valid_topk_accuracy,
            topk_accuracy_ci=total_valid_topk_accuracy_ci,
            duration=total_valid_duration,
        )
        # A couple of interesting variables to explore manually.
        valid_accuracies = [runres["eval_valid"]["accuracy"] for runres in run_results]
        valid_topk_accuracies = [runres["eval_valid"]["topk_accuracy"] for runres in run_results]
        print("-- Final stats --")
        print("Model name:", final_stats["model_name"])
        print(
            "Number of trainable model parameters:",
            final_stats["num_trainable_params"]
        )
        print("Size of an embedding vector:", final_stats["embedding_dim"])
        print("Dataset part:", final_stats["dataset_part"])
        print(
            "Number of processed labels: {0}/{1}"
                .format(final_stats["num_labels"], total_num_valid_labels)
        )
        print(
            "Number of processed images: {0}/{1}"
                .format(final_stats["num_images"], total_num_valid_images)
        )
        print(
            "Accuracy: {0:.4} +/- {1:.2} with {2:.3} confidence"
                .format(
                    final_stats["accuracy"],
                    final_stats["accuracy_ci"],
                    final_stats["accuracy_confidence"]
                )
        )
        print(
            "Top {0} accuracy: {1:.4} +/- {2:.2} with {3:.3} confidence"
                .format(
                    final_stats['topk'],
                    final_stats['topk_accuracy'],
                    final_stats["topk_accuracy_ci"],
                    final_stats["accuracy_confidence"]
                )
        )
        print("Calculation duration:", final_stats["duration"])
        print("-----------------")


    #%% Save embeddings.
    if SAVE_EMBEDDINGS:
        embeddings_filename = pathlib.Path("../data/models/") / (MODEL_NAME + ".embed")
        store.save_embeddings(
            embeddings=encoder.embeddings.cpu().numpy(),
            targes=encoder.labels,
            filename=embeddings_filename,
            inform=True,
        )


    #%% Cumulative accuracy function.
    all_label_indexes = np.arange(
        start=0, stop=evres_valid["num_labels"], step=1
    )
    mcm_valid = multilabel_confusion_matrix(
        y_true=evres_valid["targets"], y_pred=evres_valid["predictions"],
        labels=all_label_indexes
    )
    true_pos = mcm_valid[:, 1, 1]  # True positive.
    false_neg = mcm_valid[:, 1, 0]  # False negative.
    class_samples = true_pos + false_neg
    assert(np.sum(class_samples) == len(encoder.valid_dl.dataset))
    class_accuracies = true_pos / class_samples
    cum_accuracies, class_shares = cumacc.cumulative_accuracy(
        accuracies=class_accuracies, num_intervals=20
    )
    visual.draw_cumulative_accuracy(
        shares=class_shares, cum_accs=cum_accuracies, figsize=(4, 4)
    )


    #%% The most problematic classes.
    indexes_of_sorted_class_accuracies = np.argsort(class_accuracies)
    class_labels = encoder.valid_dl.dataset.get_labels()
    class_accuracy_threshold = 0.8
    least_accurate_classes = list()
    for idx in indexes_of_sorted_class_accuracies:
        if class_accuracies[idx] > class_accuracy_threshold:
            break
        least_accurate_classes.append(dict(
            label=class_labels[idx], accuracy=float(class_accuracies[idx])
        ))


    #%% Manual checking.
    num_images = len(encoder.valid_dl.dataset)
    # image_idx = 23
    image_idx = random.randrange(num_images)
    image_path = encoder.valid_dl.dataset.get_image_path(image_idx)
    image, target_label_idx = encoder.valid_dl.dataset[image_idx]
    inputs = image.unsqueeze(0)
    result = encoder.infer(inputs=inputs)
    predicted_label_indexes = [p["corpus_id"] for p in result[0]]
    target_label = encoder.valid_dl.dataset.get_label(target_label_idx)
    predicted_labels = [
        encoder.valid_dl.dataset.get_label(idx)
        for idx in predicted_label_indexes
    ]
    print("-- Manual checking --")
    print("Image index:", image_idx)
    print("Image path:", image_path)
    print("Target label:", target_label)
    print("Predicted labels:", predicted_labels)
    print("Correct:", predicted_labels[0] == target_label)
    print("Target is among predicted:", target_label in predicted_labels)
    print("---------------------")


    #%% Visual checking.
    # https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    # https://docs.pytorch.org/vision/main/auto_examples/others/plot_visualization_utils.html
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html
    num_images = len(encoder.valid_dl.dataset)
    image_indexes = np.random.permutation(num_images)[:6]
    inputs = list()
    target_labels = list()
    for image_idx in image_indexes:
        image, target_label_idx = encoder.valid_dl.dataset[image_idx]
        inputs.append(image)
        target_labels.append(encoder.valid_dl.dataset.get_label(target_label_idx))
    inputs = torch.stack(inputs)
    result = encoder.infer(inputs=inputs)
    predicted_label_indexes = [r[0]["corpus_id"] for r in result]
    predicted_labels = [
        encoder.valid_dl.dataset.get_label(idx)
        for idx in predicted_label_indexes
    ]
    visual.show_images(
        inputs=inputs,
        targets=target_labels, predictions=predicted_labels,
        mean=data_config["mean"], std=data_config["std"], figsize=(4, 4)
    )
