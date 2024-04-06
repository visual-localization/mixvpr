from os.path import join, exists
from collections import namedtuple
from scipy.io import loadmat
import os

import torchvision.transforms as T
import torch.utils.data as data


from PIL import Image
from sklearn.neighbors import NearestNeighbors

root_dir = '/pitts250k'

if not exists(root_dir):
    raise FileNotFoundError(
        'root_dir is hardcoded, please adjust to point to Pittsburgh dataset')

struct_dir = join(root_dir, 'datasets')
queries_dir = f"{root_dir}_queries_real"


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
    [folder,name] = dbIm.split("/")
    
    img_parts = name[:-4].split("_")
    split_info = int(img_parts[0])-int(folder)*1000
    
    new_index = int(split_info/250)
    new_folder = f"00{new_index}"
    new_path = os.path.join(f"{root_dir}_images_{str(folder)}",new_folder,name)
    return new_path
    
class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, input_transform=None, onlyDB=False, img_size = (480,640)):
        super().__init__()
        self.input_transform = input_transform
        if(self.input_transform is None):
            self.input_transform = default_input_transform(img_size)

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

        return {"image":img}, index

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
