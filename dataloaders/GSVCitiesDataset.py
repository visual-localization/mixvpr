# https://github.com/amaralibey/gsv-cities

import pandas as pd
from pathlib import Path
from PIL import Image
import os

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
from scipy.spatial.transform import Rotation
import utm

from .Scene import Scene
from .utils import read_depth_image,correct_intrinsic_scale

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# NOTE: Hard coded path to dataset folder 
BASE_PATH = '/gsv_cities'

if not Path(BASE_PATH).exists():
    raise FileNotFoundError(
        'BASE_PATH is hardcoded, please adjust to point to gsv_cities')

class GSVCitiesDataset(Dataset):
    def __init__(self,
                 cities=['London', 'Boston'],
                 img_per_place=4,
                 min_img_per_place=4,
                 random_sample_from_each_place=True,
                 transform=default_transform,
                 base_path=BASE_PATH,
                 img_size = (480,640)
                 ):
        super(GSVCitiesDataset, self).__init__()
        self.base_path = base_path
        self.cities = cities

        assert img_per_place <= min_img_per_place, \
            f"img_per_place should be less than {min_img_per_place}"
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform
        self.img_size = img_size
        
        # generate the dataframe contraining images metadata
        self.dataframe = self.__getdataframes()
        
        # get all unique place ids
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)
        
        # Setup intrinsics for cities
        self.intrinsics = self.setup_intrinsics()
        
    def __getdataframes(self):
        ''' 
            Return one dataframe containing
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing a DataFrame
            for each city in self.cities
        ''',
        # read the first city dataframe
        df = pd.read_csv(os.path.join(self.base_path,"Dataframes",f'{self.cities[0]}.csv'))
        df = df.sample(frac=1)  # shuffle the city dataframe
        

        # append other cities one by one
        for i in range(1, len(self.cities)):
            tmp_df = pd.read_csv(os.path.join(self.base_path,"Dataframes",f'{self.cities[i]}.csv'))

            # Now we add a prefix to place_id, so that we
            # don't confuse, say, place number 13 of NewYork
            # with place number 13 of London ==> (0000013 and 0500013)
            # We suppose that there is no city with more than
            # 99999 images and there won't be more than 99 cities
            prefix = i
            tmp_df['place_id'] = tmp_df['place_id'] + (prefix * 10**5)
            tmp_df = tmp_df.sample(frac=1)  # shuffle the city dataframe
            
            df = pd.concat([df, tmp_df], ignore_index=True)

        # keep only places depicted by at least min_img_per_place images
        res = df[df.groupby('place_id')['place_id'].transform(
            'size') >= self.min_img_per_place]
        return res.set_index('place_id')
    
    def __getitem__(self, index):
        place_id = self.places_ids[index]
        
        # get the place in form of a dataframe (each row corresponds to one image)
        place = self.dataframe.loc[place_id]
        
        # sample K images (rows) from this place
        # we can either sort and take the most recent k images
        # or randomly sample them
        if self.random_sample_from_each_place:
            place = place.sample(n=self.img_per_place)
        else:  # always get the same most recent images
            place = place.sort_values(
                by=['year', 'month', 'lat'], ascending=False)
            place = place[: self.img_per_place]
            
        scenes = []
        for i, row in place.iterrows():
            img_name = self.get_img_name(row)
            img_path = os.path.join(f"{self.base_path}_Images_{row['city_id']}",img_name)
            scene = self.proccess_image_from_name(img_name,img_path)
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
        return scenes, torch.tensor(place_id).repeat(self.img_per_place)

    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.places_ids)

    @staticmethod
    def image_loader(path):
        return Image.open(path).convert('RGB')

    @staticmethod
    def get_img_name(row):
        # given a row from the dataframe
        # return the corresponding image name

        city = row['city_id']
        
        # now remove the two digit we added to the id
        # they are superficially added to make ids different
        # for different cities
        pl_id = row.name % 10**5  #row.name is the index of the row, not to be confused with image name
        pl_id = str(pl_id).zfill(7)
        
        panoid = row['panoid']
        year = str(row['year']).zfill(4)
        month = str(row['month']).zfill(2)
        northdeg = str(row['northdeg']).zfill(3)
        lat, lon = str(row['lat']), str(row['lon'])
        name = city+'_'+pl_id+'_'+year+'_'+month+'_' + \
            northdeg+'_'+lat+'_'+lon+'_'+panoid+'.jpg'
        return name
    
    @staticmethod
    def generate_depth_path(base_path:str,img_path:str):
        return img_path.replace("Images","Depths").replace(".jpg",".png")
    
    @staticmethod
    def read_poses(name:str):
        name_split = name[:-4].split("_")
        lat,lon = float(name_split[5],name_split[6])
        northdeg = float(name_split[4])
        
        easting,northing,_,_ = utm.from_latlon(lat,lon)
        trans = [easting,0,northing]
        
        rotation_vector = [0,northdeg,0]
        rot = Rotation.from_euler('xyz',rotation_vector , degrees=True)
        quat = rot.as_quat()
        return torch.tensor(quat), torch.tensor(trans)
        
    
    def read_intrinsics(self, img_name: str, img_size=None,width=0,height=0):
        """
        Read the intrinsics of a specific image, according to its name
        """
        city = img_name.split("_")[0]
        intrinsic = self.intrinsics[city]
        fx, fy, cx, cy, W, H = intrinsic[0],intrinsic[0],intrinsic[1],intrinsic[2],width,height
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        if img_size is not None:
            K = correct_intrinsic_scale(K, img_size[0] / W, img_size[1] / H)
        return K,W,H
    
    def setup_intrinsics(self):
        intrinsics = {}
        txt_path = os.path.join(self.base_path,"intrinsics.txt")
        with open(txt_path,"r") as f:
            for line in f.readlines():
                line_split = line.strip().split(" ")
                intrinsics[line_split[0]]=[float(val) for val in line_split[1:]]
        return intrinsics
    
    def proccess_image_from_name(self,img_name:str,img_path:str)->Scene:
        # Load image
        img = self.image_loader(img_path)
        W,H = img.size
        if self.transform is not None:
            img = self.transform(img)
        
        # Load depth
        depth_path = self.generate_depth_path(self.base_path,img_path)
        print(depth_path)
        depth = read_depth_image(depth_path,self.img_size)
        
        # Load poses from name
        q,t = self.read_poses(img_name)
        
        # Load intrinsics matrix
        intrinsics_matrix,_,_ = self.read_intrinsics(img_name,self.img_size,W,H)
        return Scene.create_dict(
            img,depth,
            intrinsics_matrix,
            q,t
        )
        