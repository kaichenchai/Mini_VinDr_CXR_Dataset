import timm
from timm.data.dataset import ImageDataset



if __name__ == "__main__":
    transforms = 

    model = timm.create_model("resnet50d", pretrained=True, num_classes=2)
    # print(model.conv1[0].weight.dtype)
    print(model)

    base_path = ""
    class_map = {"normal": 0,
                 "cardiomegaly": 1}
    train_dataset = ImageDataset(root=f"{base_path}/train", split="train", class_map=class_map)
    val_dataset = ImageDataset(root=f"{base_path}/val", split="validation", class_map=class_map)


