# https://github.com/amaralibey/gsv-cities
from PIL import Image
from typing import TypedDict

import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.spatial.transform import Rotation
import utm
from datasets import load_dataset
import cv2
import torchvision.transforms as T

from .Scene import Scene
from .utils import correct_intrinsic_scale


default_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_default_transform(img_size: tuple[int, int]):
    return T.Compose(
        [
            T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class Intrinsics(TypedDict):
    fx: float
    fy: float
    cx: float
    cy: float


class Metadata(TypedDict):
    place_id: int
    city_id: str
    panoid: str
    year: int
    month: int
    northdeg: float
    lat: float
    lon: float


class Example(TypedDict):
    image: Image.Image
    depth: Image.Image
    intrinsics: Intrinsics
    metadata: dict


class GSVCitiesDataset(Dataset):
    def __init__(
        self,
        cities=None,
        min_img_per_place=4,
        transform=None,
        img_size: tuple[int, int] = (480, 640),
    ):
        if cities is None:
            cities = ["London", "Boston"]

        if transform is None:
            transform = get_default_transform(img_size)

        super(GSVCitiesDataset, self).__init__()
        self.dataset = load_dataset(
            "vpr-rpr-localization/gsv-cities",
            "CustomCities",
            token="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",
            trust_remote_code=True,
            cities=cities,
            min_img_per_place=min_img_per_place,
        )

        self.dataset = self.dataset["train"]
        self.transform = transform
        self.img_size = img_size
        self.min_img_per_place = min_img_per_place

    def __getitem__(self, index):
        examples: list[Example] = self.dataset[index]["batch"]
        place_id = examples[0]["metadata"]["place_id"]

        scenes = []
        for example in examples:
            scene = self.process_example(example)
            scenes.append(scene)

        scenes = {k: torch.stack([dic[k] for dic in scenes]) for k in scenes[0]}

        # NOTE: contrary to image classification where __getitem__ returns only one image
        # in GSVCities, we return a place, which is a Tesor of K images (K=self.img_per_place)
        # this will return a Tensor of shape [K, channels, height, width]. This needs to be taken into account
        # in the Dataloader (which will yield batches of shape [BS, K, channels, height, width])

        # NOTE: contrary to what is written above, the class returns a list of K images, which is formatted as
        # {
        #     "img": [K imgs]
        #     "depth":
        #     ...
        # }
        # Therefore in the dataloader, it will return a similar dict, but more complicated
        # {
        #     "img": [BS, K imgs]
        #     "depth":
        #     ...
        # }
        return scenes, torch.tensor(place_id).repeat(self.min_img_per_place)

    def __len__(self):
        """Denotes the total number of places (not images)"""
        return len(self.dataset)

    @staticmethod
    def read_poses(metadata: dict):
        lat = metadata["lat"]
        lon = metadata["lon"]
        northdeg = metadata["northdeg"]

        easting, northing, _, _ = utm.from_latlon(lat, lon)
        trans = [easting, 0, northing]

        rotation_vector = [0, northdeg, 0]
        rot = Rotation.from_euler("xyz", rotation_vector, degrees=True)
        quat = rot.as_quat()
        return torch.tensor(quat), torch.tensor(trans)

    def read_intrinsics(
        self, intrinsics: Intrinsics, img_size: tuple[int, int], width=0, height=0
    ):
        """
        Read the intrinsics of a specific image, according to its name
        """
        fx, fy, cx, cy, W, H = (
            intrinsics["fx"],
            intrinsics["fy"],
            intrinsics["cx"],
            intrinsics["cy"],
            width,
            height,
        )

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        if img_size is not None:
            K = correct_intrinsic_scale(K, img_size[0] / W, img_size[1] / H)

        K = torch.tensor(K, dtype=torch.float32)
        return K, W, H

    def process_example(self, example: Example) -> Scene:
        W, H = example["image"].size
        image = self.transform(example["image"].convert("RGB"))

        depth = np.array(example["depth"])
        depth = depth / 1000
        depth = torch.from_numpy(depth)  # (h, w)

        # Resizing
        trans = T.Resize(self.img_size, interpolation=T.InterpolationMode.BILINEAR)
        depth = trans(depth.unsqueeze(dim=0)).squeeze()

        # Load poses from name
        q, t = self.read_poses(example["metadata"])

        # Load intrinsics matrix
        intrinsics_matrix, _, _ = self.read_intrinsics(
            example["intrinsics"], self.img_size, W, H
        )

        return Scene.create_dict(image, depth, intrinsics_matrix, q, t)
