import torchvision.transforms as T
import torch.utils.data as data

from datasets import load_dataset


def get_input_transform(image_size=None):
    return T.Compose(
        [
            T.Resize(image_size),  # interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_whole_val_set(input_transform, img_size=(480, 640)):
    return WholeDatasetFromStruct(
        "pitts30k", "validation", input_transform=input_transform, img_size=img_size
    )


def get_250k_val_set(input_transform, img_size=(480, 640)):
    return WholeDatasetFromStruct(
        "pitts250k", "validation", input_transform=input_transform, img_size=img_size
    )


def get_whole_test_set(input_transform, img_size=(480, 640)):
    return WholeDatasetFromStruct(
        "pitts30k", "test", input_transform=input_transform, img_size=img_size
    )


def get_250k_test_set(input_transform):
    return WholeDatasetFromStruct(
        "pitts250k", "test", input_transform=input_transform, img_size=(480, 640)
    )


# TODO: add onlyDb back as params
def get_whole_training_set():
    return WholeDatasetFromStruct(
        "pitts30k", "train", input_transform=get_input_transform()
    )


class WholeDatasetFromStruct(data.Dataset):
    def __init__(
        self,
        sub_name: str,
        split: str,
        input_transform=None,
        img_size=(480, 640),
    ):
        super().__init__()

        if input_transform is None:
            input_transform = get_input_transform(img_size)

        self.dataset = load_dataset(
            "vpr-rpr-localization/pittsburgh-hf",
            sub_name,
            token="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",
            trust_remote_code=True,
        )

        self.dataset = self.dataset[split]
        self.input_transform = input_transform

    def __getitem__(self, index):
        features = self.dataset[index]
        img = features["image"]

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.dataset)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        return self.dataset[0]["positives"]
