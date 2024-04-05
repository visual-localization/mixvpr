from modal import Stub, Volume,Image,Mount

from typing import Dict
import os

from const import PITTS,GSV



def lookup_volume(data_dict:Dict[str,str]):
    return dict((k, Volume.lookup(v)) for k, v in data_dict.items())
    
stub = Stub(
    name="Im trying my best :((("
)

image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(["ffmpeg","libsm6","libxext6"])
    .pip_install_from_requirements("./requirements.txt")
)


@stub.function(
    image=image,
    mounts=[Mount.from_local_dir("./", remote_path="/root/mixvpr")],
    volumes=lookup_volume({**GSV})
)
def entry():
    import sys
    sys.path.append("/root/mixvpr")
    
    import os
    print(os.getcwd())
    print(os.listdir(os.getcwd()))
    
    import torchvision.transforms as T
    from dataloaders.GSVCitiesDataset import GSVCitiesDataset
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
    
    print(test_item[0]["image"].shape)
    print(test_item[0]["depth"].shape)
    print(test_item[0]["intrinsics_matrix"].shape)
    print(test_item[0]["rotation"].shape)
    print(test_item[0]["translation"].shape)