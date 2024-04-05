from modal import Stub, Volume,Image,Mount

from typing import Dict
import os

from const import PITTS,GSV

stub = Stub("gsv_loader")

def lookup_volume(data_dict:Dict[str,str]):
    return dict((k, Volume.lookup(v)) for k, v in data_dict.items())

image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .run_commands("git clone --single-branch --branch remote-volume https://github.com/visual-localization/mixvpr.git && cd mixvpr")
    .run_commands("pip install -r requirements.txt")
)

with image.imports():
    from dataloaders.GSVCitiesDataset import GSVCitiesDataset
    import torchvision.transforms as T

@stub.function(volumes=lookup_volume({**GSV}),image=image)
def run():
    image_size = (480,640)
    
    IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}
    
    dataset = GSVCitiesDataset(
        cities=["LosAngeles"],
        img_per_place=8,
        min_img_per_place=16,
        random_sample_from_each_place=True,
        transform=T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]),
        ]),
        img_size = image_size
    )
    test_item = dataset[0]
    print(test_item)
    
    print(test_item["image"].shape)
    print(test_item["depth"].shape)
    print(test_item["intrinsics_matrix"].shape)
    print(test_item["rotation"].shape)
    print(test_item["translation"].shape)