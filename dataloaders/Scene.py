## A datasets of images as for both DB and query should be containing:
# ##### path/to/db
#         => image: containing image files with name: cat_fx_fy_cx_cy_width_height
#         => depth: cat.dptkitti
#         => poses.txt: containing pose of the image if it is a db of reference images: cat qw qx qy qz tx ty tz

# ##### path/to/query
#         => image: containing image files with name: cat_frameid_fx_fy_cx_cy_width_height
#         => depth: cat_frameid.dptkitti
#         => poses.txt: containing pose of the image if it is a db of reference images: cat_frameid qw qx qy qz tx ty tz

import os
from pathlib import Path
from typing import Optional,Dict,Tuple,Any, Union
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

def transform(img_path, resize):
  cv_type = cv2.IMREAD_COLOR
  image = cv2.imread(img_path, cv_type)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image, resize)
  image = torch.from_numpy(image).float().permute(2, 0, 1) / 255
  return image



class Scene:
    image: torch.tensor # (ch,h,w)
    depth: torch.tensor # (h,w)
    intrinsics_matrix: torch.tensor #(3,3)
    rotation: torch.tensor #[4]: q1,q2,q3,q4
    translation: torch.tensor # [3]: x,y,z: should be the absolute pose of the image
    def create_dict(
        image:torch.tensor, # (h, w, 3) in cpu
        depth:torch.tensor,
        intrinsics_matrix: torch.tensor,
        rotation: torch.tensor,
        translation: torch.tensor
    ):
        return{
            "image": image,
            "depth": depth,
            "intrinsics_matrix":intrinsics_matrix,
            "rotation": rotation,
            "translation":translation,
        }