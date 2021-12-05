import cv2
import config
import tensorflow as tf
import numpy as np

def get_image(path):
    image=cv2.imread(path,cv2.IMREAD_COLOR)
    image = cv2.resize(image,(config.width,config.height))
    image=image/255
    image = image.astype(np.float32)
    return image

def get_mask(path):
    mask = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(config.width,config.height))
    mask = mask.astype(np.int32)

    return mask

def preprocess_image_and_mask(image_path,mask_path):
    def get_data(image_path, mask_path):
        image_path=str(image_path,'utf-8')
        mask_path=str(mask_path,'utf-8')
        image = get_image(image_path)
        mask = get_mask(mask_path)

        return image,mask

    image, mask=tf.numpy_function(get_data,[image_path,mask_path],[tf.float32,tf.int32])
    mask = tf.one_hot(mask, config.num_classes, dtype=tf.float32)
    image.set_shape([config.width,config.height, 3])
    mask.set_shape([config.width,config.height, config.num_classes])
    return image,mask

def dataset(image_paths,mask_paths):
    dataset=tf.data.Dataset.from_tensor_slices((image_paths,mask_paths))
    dataset=dataset.map(preprocess_image_and_mask,num_parallel_calls=tf.data.AUTOTUNE).batch(config.batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return dataset