import numpy as np


def save_point_cloud_as_ply(save_path:str, pcds:np.ndarray, rgbs:np.ndarray=None) -> None:
    """
    Save point cloud as ply.

    Args:
        save_path (str)
        pcds (np.ndarray)
        rgbs (np.ndarray, optional)
    """

    pcds = pcds.reshape((-1,3))
    if rgbs is not None:
        rgbs = rgbs.reshape((-1,3))
        
    header = (
        'ply\n'
        'format ascii 1.0\n'
        f'element vertex {len(pcds)}\n'
        'property float x\n'
        'property float y\n'
        'property float z\n'
    )
    
    if rgbs is not None:
        header += (
            'property uint8 red\n'
            'property uint8 green\n'
            'property uint8 blue\n'
        )
        
    header += 'end_header\n'

    with open(save_path, 'w') as f:
        
        f.write(header)
        
        if rgbs is None:
            np.savetxt(f, pcds, '%f %f %f', delimiter=' ')
        else:
            tmp = np.concatenate((pcds, rgbs), axis=-1)
            np.savetxt(f, tmp, '%f %f %f %d %d %d', delimiter=' ')


def convert_depth_to_point_cloud(depth:np.ndarray, intrinsic:np.ndarray, scale:float=1.0) -> np.ndarray:
    """
    Convert depth image(s) to 3D point cloud(s) using camera intrinsics.

    Args:
        depth (np.ndarray)
        intrinsic (np.ndarray)
        scale (float, optional): scaling factor to convert depth values to meters. Defaults to 1.0.

    Returns:
        np.ndarray: point cloud(s)
    """

    if depth.ndim == 2:
        batched_input = False
        depth = depth[None,...]
        intrinsic = intrinsic[None,...]
    elif depth.ndim == 3:
        batched_input = True
        if intrinsic.ndim == 2:
            intrinsic = intrinsic[None,...]
    else:
        raise NotImplementedError
        
    H, W = depth.shape[1], depth.shape[2]
    u,v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    depth = depth.astype(np.float32)
    intrinsic = intrinsic.astype(np.float32)
    
    Z = depth / scale
    X = (u[None,...] - intrinsic[:,0,2][:,None,None]) * Z / intrinsic[:,0,0][:,None,None]
    Y = (v[None,...] - intrinsic[:,1,2][:,None,None]) * Z / intrinsic[:,1,1][:,None,None]
    
    point_cloud = np.stack((X, Y, Z), axis=-1)
    
    if batched_input:
        return point_cloud
    else:
        return point_cloud[0]
    

def transform_point_cloud(point_cloud:np.ndarray, RT:np.ndarray) -> np.ndarray:
    """
    Trasform point cloud(s) using rigid transformation matrix/matrices.

    Args:
        point_cloud (np.ndarray)
        RT (np.ndarray): rigid transformation matrix/matrices
    Returns:
        np.ndarray: transformed point cloud(s)
    """
    
    if RT.ndim == 2:
        original_shape = point_cloud.shape
        point_cloud = point_cloud.reshape((-1,3))
        point_cloud_homogeneous = np.concatenate((point_cloud, np.ones_like(point_cloud[...,:1])), axis=-1)
        point_cloud_homogeneous = point_cloud_homogeneous @ RT.T
        transformed_point_cloud = point_cloud_homogeneous[...,:3]
        transformed_point_cloud = transformed_point_cloud.reshape(original_shape)
    elif RT.ndim == 3:
        assert point_cloud.shape[0] == RT.shape[0], "Batch size of point cloud and transformation matrix must match."
        assert point_cloud.ndim >= 3, "Point cloud must have at least 3 dimensions (B,N,3)."
        original_shape = point_cloud.shape
        point_cloud = point_cloud.reshape((point_cloud.shape[0], -1, 3))
        point_cloud_homogeneous = np.concatenate((point_cloud, np.ones_like(point_cloud[...,:1])), axis=-1)
        point_cloud_homogeneous = np.matmul(point_cloud_homogeneous, RT.transpose(0, 2, 1))
        transformed_point_cloud = point_cloud_homogeneous[...,:3]
        transformed_point_cloud = transformed_point_cloud.reshape(original_shape)
        
    return transformed_point_cloud
