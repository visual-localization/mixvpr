from os.path import join, exists
from collections import namedtuple
from scipy.io import loadmat
import os

import torchvision.transforms as T
import torch.utils.data as data


from PIL import Image
from sklearn.neighbors import NearestNeighbors

root_dir = '../datasets/Pittsburgh/'

if not exists(root_dir):
    raise FileNotFoundError(
        'root_dir is hardcoded, please adjust to point to Pittsburgh dataset')

struct_dir = join(root_dir, 'datasets/')
queries_dir = join(root_dir, 'queries_real')


def input_transform(image_size=None):
    return T.Compose([
        T.Resize(image_size),# interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



def get_whole_val_set(input_transform):
    structFile = join(struct_dir, 'pitts30k_val.mat')
    return WholeDatasetFromStruct(structFile, input_transform=input_transform)


def get_250k_val_set(input_transform):
    structFile = join(struct_dir, 'pitts250k_val.mat')
    return WholeDatasetFromStruct(structFile, input_transform=input_transform)


def get_whole_test_set(input_transform):
    structFile = join(struct_dir, 'pitts30k_test.mat')
    return WholeDatasetFromStruct(structFile, input_transform=input_transform)


def get_250k_test_set(input_transform):
    structFile = join(struct_dir, 'pitts250k_test.mat')
    return WholeDatasetFromStruct(structFile, input_transform=input_transform)

def get_whole_training_set(onlyDB=False):
    structFile = join(struct_dir, 'pitts30k_train.mat')
    return WholeDatasetFromStruct(structFile,
                                  input_transform=input_transform(),
                                  onlyDB=onlyDB)

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
    [folder,name] = dbIm.split("/")
    
    img_parts = name[:-4].split("_")
    split_info = int(img_parts[0])-int(folder)*1000
    
    new_index = int(split_info/250)
    new_folder = f"00{new_index}"
    new_path = os.path.join(root_dir,str(folder),new_folder,name)
    return new_path
    
class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, input_transform=None, onlyDB=False, img_size = (480,640)):
        super().__init__()

        self.input_transform = input_transform

        self.dbStruct = parse_dbStruct(structFile)
        self.images = [join_db_img(root_dir, dbIm) for dbIm in self.dbStruct.dbImage]
        if not onlyDB:
            self.images += [join(queries_dir, qIm)
                            for qIm in self.dbStruct.qImage]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None

    def __getitem__(self, index):
        img = Image.open(self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

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
# from os.path import join, exists
# from collections import namedtuple
# from scipy.io import loadmat
# import numpy as np

# import torchvision.transforms as T
# import torch.utils.data as data


# from PIL import Image
# from sklearn.neighbors import NearestNeighbors

# from .Scene import Scene
# from .utils import read_depth_image,correct_intrinsic_scale

# root_dir = '../datasets/Pittsburgh/'

# if not exists(root_dir):
#     raise FileNotFoundError(
#         'root_dir is hardcoded, please adjust to point to Pittsburgh dataset')

# struct_dir = join(root_dir, 'datasets/')
# queries_dir = join(root_dir, 'queries_real')
# images_dir = join(root_dir, "images/")

# def input_transform(image_size=None):
#     return T.Compose([
#         T.Resize(image_size),# interpolation=T.InterpolationMode.BICUBIC),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])



# def get_whole_val_set(input_transform,img_size):
#     structFile = join(struct_dir, 'pitts30k_val.mat')
#     return WholeDatasetFromStruct(structFile, input_transform=input_transform,img_size=img_size)


# def get_250k_val_set(input_transform,img_size):
#     structFile = join(struct_dir, 'pitts250k_val.mat')
#     return WholeDatasetFromStruct(structFile, input_transform=input_transform,img_size=img_size)


# def get_whole_test_set(input_transform,img_size):
#     structFile = join(struct_dir, 'pitts30k_test.mat')
#     return WholeDatasetFromStruct(structFile, input_transform=input_transform,img_size=img_size)


# def get_250k_test_set(input_transform,img_size):
#     structFile = join(struct_dir, 'pitts250k_test.mat')
#     return WholeDatasetFromStruct(structFile, input_transform=input_transform,img_size=img_size)

# def get_whole_training_set(onlyDB=False,img_size=(320,320)):
#     structFile = join(struct_dir, 'pitts30k_train.mat')
#     return WholeDatasetFromStruct(structFile,
#                                   input_transform=input_transform(),
#                                   onlyDB=onlyDB,img_size=img_size)

# dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset',
#                                    'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
#                                    'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])


# def parse_dbStruct(path):
#     mat = loadmat(path)
#     matStruct = mat['dbStruct'].item()

#     if '250k' in path.split('/')[-1]:
#         dataset = 'pitts250k'
#     else:
#         dataset = 'pitts30k'

#     whichSet = matStruct[0].item()

#     dbImage = [f[0].item() for f in matStruct[1]]
#     utmDb = matStruct[2].T

#     qImage = [f[0].item() for f in matStruct[3]]
#     utmQ = matStruct[4].T

#     numDb = matStruct[5].item()
#     numQ = matStruct[6].item()

#     posDistThr = matStruct[7].item()
#     posDistSqThr = matStruct[8].item()
#     nonTrivPosDistSqThr = matStruct[9].item()

#     return dbStruct(whichSet, dataset, dbImage, utmDb, qImage,
#                     utmQ, numDb, numQ, posDistThr,
#                     posDistSqThr, nonTrivPosDistSqThr)


# class WholeDatasetFromStruct(data.Dataset):
#     def __init__(self, structFile, input_transform=None, onlyDB=False, img_size = (480,640)):
#         super().__init__()

#         self.input_transform = input_transform

#         self.dbStruct = parse_dbStruct(structFile)
#         self.images = [join(images_dir, dbIm) for dbIm in self.dbStruct.dbImage]
#         if not onlyDB:
#             self.images += [join(queries_dir, qIm)
#                             for qIm in self.dbStruct.qImage]

#         self.whichSet = self.dbStruct.whichSet
#         self.dataset = self.dbStruct.dataset

#         self.positives = None
#         self.distances = None
        
#         self.img_size = img_size
#         self.base_path = root_dir

#     @staticmethod
#     def read_poses(name:str):
#         #TODO: Implement this shit
#         pass
    
#     def read_intrinsics(self, img_name: str, img_size=None,width=0,height=0):
#         """
#         Read the intrinsics of a specific image, according to its name
#         """
#         fx, fy, cx, cy, W, H = 768.000000, 768.00000, 320.000000, 240.000000, 640.0, 480.0
#         K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
#         if img_size is not None:
#             K = correct_intrinsic_scale(K, img_size[0] / W, img_size[1] / H)
#         return K,W,H
    
#     @staticmethod
#     def generate_depth_path(base_path:str,img_path:str):
#         src_arr = ["images","queries_real"]
#         for src in src_arr:
#             if(src in img_path):
#                 return img_path.replace(src,src+"_depth")
    
#     def __getitem__(self, index):
#         img_path = self.images[index]
#         img_name = img_path[len(img_path):]
#         img = Image.open(img_path)
#         W,H = img.size
#         if self.input_transform:
#             img = self.input_transform(img)
        
#         # Load depth
#         depth_path = self.generate_depth_path(self.base_path,img_path)
#         depth = read_depth_image(depth_path,self.img_size)
        
#         # Load poses from name
#         q,t = self.read_poses(img_name)
        
#         # Load intrinsics matrix
#         intrinsics_matrix,_,_ = self.read_intrinsics(img_name="",img_size=self.img_size,width=W,height=H)
#         return Scene.create_dict(
#             img,depth,
#             intrinsics_matrix,
#             q,t
#         ),index

#     def __len__(self):
#         return len(self.images)

#     def getPositives(self):
#         # positives for evaluation are those within trivial threshold range
#         # fit NN to find them, search by radius
#         if self.positives is None:
#             knn = NearestNeighbors(n_jobs=-1)
#             knn.fit(self.dbStruct.utmDb)

#             self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ,
#                                                                   radius=self.dbStruct.posDistThr)

#         return self.positives
