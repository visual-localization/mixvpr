import torch
from typing import Dict

from typing import Tuple

device='cuda' if torch.cuda.is_available() else 'cpu'


def set_device_tensor(img:Dict[str,torch.Tensor]):
    return {
        key:value.to(device) if value.get_device()==-1 else value for key,value in img.items()
    }

def frustum_difference(origin_img,target_img)->float:
    origin_img = set_device_tensor(origin_img)
    target_img = set_device_tensor(target_img)
    
    intrinsics_matrix = origin_img["intrinsics_matrix"].double()
    inv_intrinsics = torch.inverse(intrinsics_matrix)
    
    depth = origin_img["depth"].double()
    height = depth.shape[0]
    width  = depth.shape[1]
    
    interval = 10
    height_grid = torch.arange(0,height,interval,dtype=torch.long).to(device=device)
    width_grid  = torch.arange(0,width,interval,dtype=torch.long).to(device=device)
    grid = torch.cartesian_prod(width_grid, height_grid) 

    depth_grid = depth[grid[:,1],grid[:,0]]

    grid_cam = torch.cat([grid,torch.ones(grid.shape[0]).reshape(-1,1).to(device=device)],1).double()
    rev_intrinsics_grid = inv_intrinsics @ torch.transpose(grid_cam,0,1)
    camera_coord_grid = torch.transpose(rev_intrinsics_grid,0,1) * depth_grid.reshape(-1,1)

    mat1 = quat2mat_custom(origin_img["rotation"]).double()
    trans1 = origin_img["translation"].double()

    abs_point = torch.transpose(mat1@torch.transpose(camera_coord_grid,0,1),0,1) + trans1

    mat2 = quat2mat_custom(target_img["rotation"]).double()
    trans2 = target_img["translation"].double()
    cam2_intrinsics = target_img["intrinsics_matrix"].double()

    point_in_query = torch.inverse(mat2) @ torch.transpose(abs_point-trans2,0,1)
    point_in_query_2 = cam2_intrinsics@point_in_query

    temp = torch.transpose(point_in_query_2,0,1)
    target_points = temp/temp[:,2].reshape(-1,1)

    return torch.sum((target_points[:,0]>0) & (target_points[:,1]>0) & (target_points[:,0]<depth.shape[1]) & (target_points[:,1]<depth.shape[0]))/target_points.shape[0]

def angle_difference(q1:torch.tensor,q2:torch.tensor):
    mat1 = quat2mat_custom(q1)
    mat2 = quat2mat_custom(q2)
    PI = torch.acos(torch.zeros(1).to(device=device)).item() * 2
    return ((torch.trace(torch.transpose(mat1,0,1)@mat2)-1)/2)*(180/PI)

def quat2mat_custom(q:torch.tensor):
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < 2.220446049250313e-16:
        return torch.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return torch.tensor(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]]).double().to(device=device)