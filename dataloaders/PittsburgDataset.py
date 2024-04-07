from os.path import join, exists
from collections import namedtuple
from scipy.io import loadmat
import os

import torchvision.transforms as T
import torch.utils.data as data
import numpy as np
import torch

from PIL import Image
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Tuple

from .Scene import Scene
from .utils import read_depth_image,correct_intrinsic_scale

root_dir = '/pitts250k'

if not exists(root_dir):
    raise FileNotFoundError(
        'root_dir is hardcoded, please adjust to point to Pittsburgh dataset')

struct_dir = join(root_dir, 'datasets')
queries_dir = f"{root_dir}_queries_real"
pose_dir = join(root_dir,"poses")


def default_input_transform(image_size=None):
    return T.Compose([
        T.Resize(image_size),# interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



def get_whole_val_set(input_transform,img_size):
    structFile = join(struct_dir, 'pitts30k_val.mat')
    return WholeDatasetFromStruct(
        structFile,
        input_transform=input_transform,
        img_size=img_size
    )


def get_250k_val_set(input_transform,img_size):
    structFile = join(struct_dir, 'pitts250k_val.mat')
    return WholeDatasetFromStruct(
        structFile,
        input_transform=input_transform,
        img_size=img_size
    )


def get_whole_test_set(input_transform,img_size):
    structFile = join(struct_dir, 'pitts30k_test.mat')
    return WholeDatasetFromStruct(
        structFile,
        input_transform=input_transform,
        img_size=img_size
    )


def get_250k_test_set(input_transform,img_size):
    structFile = join(struct_dir, 'pitts250k_test.mat')
    return WholeDatasetFromStruct(
        structFile,
        input_transform=input_transform,
        img_size=img_size
    )

def get_whole_training_set(input_transform,img_size,onlyDB=False):
    structFile = join(struct_dir, 'pitts30k_train.mat')
    return WholeDatasetFromStruct(
        structFile,
        input_transform=input_transform,
        onlyDB=onlyDB,
        img_size=img_size
    )

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset',
                                   'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
                                   'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])


def parse_dbStruct(path):
    mat = loadmat(path)
    matStruct = mat['dbStruct'].item()

    if '250k' in path.split('/')[-1]:
        dataset = 'pitts250k'
    else:
        dataset = 'pitts30k'

    whichSet = matStruct[0].item()

    dbImage = [f[0].item() for f in matStruct[1]]
    utmDb = matStruct[2].T

    qImage = [f[0].item() for f in matStruct[3]]
    utmQ = matStruct[4].T

    numDb = matStruct[5].item()
    numQ = matStruct[6].item()

    posDistThr = matStruct[7].item()
    posDistSqThr = matStruct[8].item()
    nonTrivPosDistSqThr = matStruct[9].item()

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage,
                    utmQ, numDb, numQ, posDistThr,
                    posDistSqThr, nonTrivPosDistSqThr)

def join_db_img(root_dir,dbIm):
    [folder,name] = dbIm.split("/") #003/0003313
    
    img_parts = name[:-4].split("_")
    split_info = int(img_parts[0])-int(folder)*1000
    
    new_index = int(split_info/250)
    new_folder = f"00{new_index}" #000/001/002/003
    new_path = os.path.join(f"{root_dir}_images_{str(folder)}",new_folder,name)
    return new_path

def alt_convert_zxy(point_xyz):
    point_xyz[0], point_xyz[1], point_xyz[2] = point_xyz[0], -point_xyz[2], point_xyz[1]
    return point_xyz

class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, input_transform=None, onlyDB=False, img_size = (480,640)):
        super().__init__()
        
        dataset_name = structFile[:-4].split('/')[-1]
        
        self.input_transform = input_transform
        self.image_size = img_size
        if(self.input_transform is None):
            self.input_transform = default_input_transform(img_size)

        self.dbStruct = parse_dbStruct(structFile)
        self.images_path = [join_db_img(root_dir, dbIm) for dbIm in self.dbStruct.dbImage]
        self.images_name = [dbIm for dbIm in self.dbStruct.dbImage]
        self.poses = self.read_poses("db",dataset_name)
        if not onlyDB:
            self.images_path += [join(queries_dir, qIm)
                                for qIm in self.dbStruct.qImage]
            self.images_name += [qIm
                                for qIm in self.dbStruct.qImage]
            self.poses = {
                **self.poses,
                **self.read_poses("query",dataset_name)
            }

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None
    
    @staticmethod
    def read_intrinsics(img_name: str, resize=None):
        """
        Read the intrinsics of a specific image, according to its name
        """
        fx, fy, cx, cy, W, H = 768.000, 768.000, 320, 240, 648, 480
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        if resize is not None:
            K = correct_intrinsic_scale(K, resize[0] / W, resize[1] / H)
        return K,W,H


    def generate_depth_path(self,img_path:str)->str:
        if 'queries' in str(img_path):
            depth_path=img_path.replace("queries_real","queries_depths")[:-4]
        else:
            depth_path=img_path.replace("images","depths")[:-4]
        return depth_path+".png"

    @staticmethod
    def read_poses(mode:str,struct_name:str) -> Dict[str,Tuple[np.ndarray,np.ndarray]]:
        """
        Returns a dictionary that maps: img_path -> (q, t) where
        np.array q = (qw, qx qy qz) quaternion encoding rotation matrix;
        np.array t = (tx ty tz) translation vector;
        (q, t) encodes absolute pose (world-to-camera), i.e. X_c = R(q) X_W + t
        """
        filename = join(pose_dir,f"{struct_name}_{mode}.txt")
        # filename = "pitts250k_test_query.txt"  if mode == "query"         #     else "pitts250k_test_db.txt"
        poses = {}
        with (filename).open('r') as f:
            for line in f.readlines():
                if(".jpg" not in line):
                    continue
                line = line.strip().split(" ")
                img_name = line[0]
                qt = torch.tensor(list(map(float, line[1:])))
                poses[img_name] = (qt[3:],alt_convert_zxy(qt[:3]))
        return poses

    def __getitem__(self, index):
        img_path = self.images_path[index]
        img_name = self.images_name[index]
        img = Image.open(img_path)
        
        if self.input_transform:
            img = self.input_transform(img)
        
        depth_path = self.generate_depth_path(img_path)
        depth = read_depth_image(depth_path,self.image_size)
        intrinsics_matrix = self.read_intrinsics(self.images[index],self.image_size)
        q,t = self.poses[img_name] 
        return Scene.create_dict(
            img,depth,
            intrinsics_matrix,
            q,t
        ), index

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.dbStruct.utmDb)

            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ,
                                                                  radius=self.dbStruct.posDistThr)

        return self.positives