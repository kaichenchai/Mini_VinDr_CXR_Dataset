import argparse
from functools import partial
from pathlib import Path

import pandas as pd
import torch
import torchmetrics
from PIL import Image
from timm import create_model
from timm.data import ImageDataset
from timm.scheduler import CosineLRScheduler
from torch import nn
from torch.utils.data import Dataset, random_split
from torchvision import transforms

from pytorch_accelerated.callbacks import (
    TrainerCallback,
    LogMetricsCallback,
    PrintProgressCallback,
    TerminateOnNaNCallback,
    MoveModulesToDeviceCallback,
    EarlyStoppingCallback,
    SaveBestModelCallback,
)
from pytorch_accelerated.finetuning import ModelFreezer
from pytorch_accelerated.trainer import (
    TrainerPlaceholderValues,
    TrainerWithTimmScheduler,
)

def load_dataset(dataset_name, dataset_path, train_transforms, eval_transforms):
    dataset_path = Path(dataset_path)

    train_dataset = ImageDataset(
        str(dataset_path / "train"), transform=train_transforms
    )
    eval_dataset = ImageDataset(
        str(dataset_path / "validation"), transform=eval_transforms
    )

    return train_dataset, eval_dataset


class AccuracyCallback(TrainerCallback):
    def __init__(self, num_classes):
        self.accuracy = torchmetrics.Accuracy(num_classes=num_classes)

    def _move_to_device(self, trainer):
        self.metrics.to(trainer.device)

    def on_training_run_start(self, trainer, **kwargs):
        self._move_to_device(trainer)

    def on_evaluation_run_start(self, trainer, **kwargs):
        self._move_to_device(trainer)

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        preds = batch_output["model_outputs"].argmax(dim=-1)
        self.accuracy.update(preds, batch[1])

    def on_eval_epoch_end(self, trainer, **kwargs):
        trainer.run_history.update_metric("accuracy", self.accuracy.compute().item())
        self.accuracy.reset()


def create_transforms(greyscale, num_channels, pretrained, dataset_name, training=True):
    if training:
        tfms = [transforms.RandomResizedCrop(224, scale=(0.5, 1.0))]

        if greyscale:
            print("adding greyscale tfm")
            tfms.append(transforms.Grayscale(num_output_channels=num_channels))

        tfms.extend([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    else:
        tfms = [transforms.Resize((224, 224))]

        if greyscale:
            print("adding greyscale tfm")
            tfms.append(transforms.Grayscale(num_output_channels=num_channels))

        tfms.append(transforms.ToTensor())

    if num_channels == 3 and pretrained:
        tfms.append(
            transforms.Normalize(
                torch.tensor([0.4850, 0.4560, 0.4060]),
                torch.tensor([0.2290, 0.2240, 0.2250]),
            )
        )
    elif num_channels == 1 and pretrained:
        tfms.append(
            transforms.Normalize(
                torch.tensor((0.4850 + 0.4560 + 0.4060) / 3),
                torch.tensor((0.2290 + 0.2240 + 0.2250) / 3),
            )
        )

    tfm_pipeline = transforms.Compose(tfms)

    print(tfm_pipeline)

    return tfm_pipeline


def create_scheduler():
    return partial(
        CosineLRScheduler,
        t_initial=TrainerPlaceholderValues.NUM_EPOCHS,
        cycle_decay=0.5,
        lr_min=1e-6,
        t_in_epochs=True,
        cycle_limit=1,
    )


def main(
    data_dir,
    num_epochs,
    frozen_lr,
    batch_size,
    dataset_name,
    num_channels,
    greyscale,
    pretrained,
    freeze,
):

    data_dir = Path(data_dir)

    num_classes = 2

    # Create model (from timm)
    model = create_model(
        "resnetrs50",
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=num_channels,
    )

    train_transforms = create_transforms(
        greyscale, num_channels, pretrained, dataset_name, training=True
    )
    eval_transforms = create_transforms(
        greyscale, num_channels, pretrained, dataset_name, training=False
    )

    # Create datasets
    train_dataset, eval_dataset = load_dataset(
        dataset_name, data_dir, train_transforms, eval_transforms
    )

    # Define loss function
    loss_func = nn.CrossEntropyLoss()

    freezer = ModelFreezer(model, freeze_batch_norms=False)
    if freeze:
        freezer.freeze()
    else:
        num_epochs = num_epochs * 2

    # Define optimizer
    optimizer = torch.optim.AdamW(freezer.get_trainable_parameters(), lr=frozen_lr)

    early_stop_counter = 3 if freeze else 6

    trainer = TrainerWithTimmScheduler(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        callbacks=[
            AccuracyCallback(num_classes=num_classes),
            MoveModulesToDeviceCallback,
            TerminateOnNaNCallback,
            PrintProgressCallback,
            LogMetricsCallback,
            SaveBestModelCallback,
            EarlyStoppingCallback(early_stopping_patience=early_stop_counter),
        ],
    )

    scheduler = create_scheduler()

    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=num_epochs,
        per_device_batch_size=batch_size,
        create_scheduler_fn=scheduler,
    )

    if freeze:
        param_groups = freezer.unfreeze()

        unfrozen_lr = frozen_lr / 10

        for param_group in optimizer.param_groups:
            param_group["lr"] = unfrozen_lr

        for idx, param_group in param_groups.items():
            optimizer.add_param_group(param_group)

        scheduler = create_scheduler()

        trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            num_epochs=num_epochs,
            per_device_batch_size=batch_size,
            create_scheduler_fn=scheduler,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="The data folder on disk.")
    parser.add_argument(
        "--epochs", required=True, help="The number of epochs to train for", type=int
    )
    parser.add_argument(
        "--lr",
        required=False,
        help="classification head lr",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--batch-size", required=False, help="batch_size", type=int, default=32
    )

    parser.add_argument(
        "--dataset", required=False, help="dataset name", type=str, default="beans"
    )
    parser.add_argument(
        "--num_chans",
        required=False,
        help="the number of image channels",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--greyscale", type=lambda s: s.lower() in ["true", "t", "yes", "1"]
    )
    parser.add_argument(
        "--pretrained", type=lambda s: s.lower() in ["true", "t", "yes", "1"]
    )
    parser.add_argument(
        "--freeze", type=lambda s: s.lower() in ["true", "t", "yes", "1"]
    )
    args = parser.parse_args()
    main(
        args.data_dir,
        args.epochs,
        args.lr,
        args.batch_size,
        args.dataset,
        args.num_chans,
        args.greyscale,
        args.pretrained,
        args.freeze,
    )
