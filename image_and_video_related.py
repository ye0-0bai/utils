import numpy as np
import imageio.v3 as iio

def read_images(image_dir):
    image_names = []
    for tmp in os.listdir(image_dir):
        if tmp.endswith(('jpg', 'png')):
            image_names.append(tmp)
    image_names = sorted(image_names)
    images = []
    for image_name in image_names:
        images.append(iio.imread(os.path.join(image_dir, image_name)))
    return np.array(images)
