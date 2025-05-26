import numpy as np

def save_point_cloud_as_ply(save_path:str, pcd:np.ndaeeay, rgb:np.ndaeeay=None) -> None:
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
