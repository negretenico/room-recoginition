from dataclasses import dataclass


@dataclass
class DatasetSplit:
    train_image_path: list
    val_image_path: list
    train_image_label: list
    val_image_label: list
