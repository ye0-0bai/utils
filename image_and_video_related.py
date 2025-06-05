import numpy as np
import imageio.v3 as iio

def read_images(image_dir, sort_key=None):
    image_names = []
    for tmp in os.listdir(image_dir):
        if tmp.endswith(('jpg', 'png')):
            image_names.append(tmp)
    image_names.sort(key=sort_key)
    images = []
    for image_name in image_names:
        images.append(iio.imread(os.path.join(image_dir, image_name)))
    return np.array(images)
