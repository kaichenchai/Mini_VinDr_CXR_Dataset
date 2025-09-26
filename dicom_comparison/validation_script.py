import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from timm import create_model
from timm.data import ImageDataset
from torchmetrics.classification import Accuracy, ConfusionMatrix
import wandb
import matplotlib.pyplot as plt
import seaborn as sns


def create_transforms(greyscale, num_channels, pretrained):
    tfms = [transforms.Resize((224, 224))]

    if greyscale:
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

    return transforms.Compose(tfms)


@torch.no_grad()
def evaluate(model, dataloader, device, class_names=None):
    model.eval()
    accuracy = Accuracy(task="binary").to(device)
    confmat = ConfusionMatrix(task="binary", num_classes=2).to(device)

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=-1)

        accuracy.update(preds, labels)
        confmat.update(preds, labels)

    final_accuracy = accuracy.compute().item()
    cm = confmat.compute().cpu().numpy()

    return final_accuracy, cm


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def main(
    model_checkpoint,
    data_dir,
    batch_size,
    num_channels,
    greyscale,
    pretrained,
    run_name
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(data_dir)

    num_classes = 2
    class_names = ["Negative", "Positive"]

    # Create model architecture
    model = create_model(
        "resnetrs50",
        pretrained=False,
        num_classes=num_classes,
        in_chans=num_channels,
    )
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.to(device)

    # Setup transforms and dataset
    eval_transforms = create_transforms(greyscale, num_channels, pretrained)
    eval_dataset = ImageDataset(str(data_dir / "val"), transform=eval_transforms)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Start wandb run
    wandb_run = wandb.init(
        project="dicom_comparison",
        name=run_name or "validation_run",
        config={
            "batch_size": batch_size,
            "num_channels": num_channels,
            "greyscale": greyscale,
            "pretrained": pretrained,
            "checkpoint": model_checkpoint,
        },
    )

    acc, cm = evaluate(model, eval_loader, device, class_names=class_names)

    # Plot and log confusion matrix
    fig = plot_confusion_matrix(cm, class_names=class_names)
    wandb_run.log({
        "validation_accuracy": acc,
        "confusion_matrix": wandb.Image(fig)
    })

    print(f"Validation Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", required=True, help="Dataset directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--greyscale", type=lambda s: s.lower() in ["true", "1", "yes"])
    parser.add_argument("--pretrained", type=lambda s: s.lower() in ["true", "1", "yes"])
    parser.add_argument("--run_name", type=str, help="wandb run name", default=None)

    args = parser.parse_args()
    main(
        model_checkpoint=args.model_checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_channels=args.num_channels,
        greyscale=args.greyscale,
        pretrained=args.pretrained,
        run_name=args.run_name,
    )
