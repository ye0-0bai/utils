import numpy as np

def save_point_cloud_as_ply(save_path:str, pcd:np.ndarray, rgb:np.ndarray=None) -> None:
    '''
    pcd: np.ndarray of shape (...,3)
    
    rgb(optional): np.ndarray of shape (...,3)
    '''

    pcd = pcd.reshape((-1,3))
    if rgb is not None:
        rgb = rgb.reshape((-1,3))
        
    header = (
        'ply\n'
        'format ascii 1.0\n'
        f'element vertex {len(pcd)}\n'
        'property float x\n'
        'property float y\n'
        'property float z\n'
    )
    
    if rgb is not None:
        header += (
            'property uint8 red\n'
            'property uint8 green\n'
            'property uint8 blue\n'
        )
        
    header += 'end_header\n'

    with open(save_path, 'w') as f:
        
        f.write(header)
        
        if rgb is None:
            np.savetxt(f, pcd, '%f %f %f', delimiter=' ')
        else:
            tmp = np.concatenate((pcd, rgb), axis=-1)
            np.savetxt(f, tmp, '%f %f %f %d %d %d', delimiter=' ')

def convert_depth_to_point_cloud(depth:np.ndarray, intrinsic:np.ndarray, scale:int=1) -> np.ndarray:
    """Convert depth image(s) to 3D point cloud(s) using camera intrinsics.

    Args:
        depth (np.ndarray)
        intrinsic (np.ndarray)
        scale (int, optional): Scaling factor to convert depth values to meters. Defaults to 1.

    Returns:
        np.ndarray: point cloud(s)
    """
    
    batched_input = True
    
    if depth.ndim == 2:
        batched_input = False
        depth = depth[None,...]
        intrinsic = intrinsic[None,...]
        
    H, W = depth.shape[1], depth.shape[2]
    u,v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    depth = depth.astype(np.float32)
    intrinsic = intrinsic.astype(np.float32)
    
    Z = depth / scale
    X = (u[None,...] - intrinsic[:,0,2][:,None,None]) * Z / intrinsic[:,0,0][:,None,None]
    Y = (v[None,...] - intrinsic[:,1,2][:,None,None]) * Z / intrinsic[:,1,1][:,None,None]
    
    point_clouds = np.stack((X, Y, Z), axis=-1)
    
    if batched_input:
        return point_clouds
    else:
        return point_clouds[0]
