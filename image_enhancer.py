import os
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"


class ImageEnchancer():
    def __init__(self) -> None:
        self.model = hub.load('models/tf_img_enchancer')
    
    def preprocess_image(self,image_path):
        """ Loads image from path and preprocesses to make it model ready
            Args:
                image_path: Path to the image file
        """
        hr_image = tf.convert_to_tensor(image_path, dtype=tf.uint8)
        # If PNG, remove the alpha channel. The model only supports
        # images with 3 color channels.
        if hr_image.shape[-1] == 4:
            hr_image = hr_image[...,:-1]
        hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
        hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
        hr_image = tf.cast(hr_image, tf.float32)
        return tf.expand_dims(hr_image, 0)

    def save_image(self,image, filename):
        """
            Saves unscaled Tensor Images.
            Args:
            image: 3D image tensor. [height, width, channels]
            filename: Name of the file to save.
        """
        if not isinstance(image, Image.Image):
            image = tf.clip_by_value(image, 0, 255)
            image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
        image.save("%s.jpg" % filename)
        print("Saved as %s.jpg" % filename)
    
    def plot_image(self,image, title=""):
        """
            Plots images from image tensors.
            Args:
            image: 3D image tensor. [height, width, channels].
            title: Title to display in the plot.
        """
        image = np.asarray(image)
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
        plt.imshow(image)
        plt.axis("off")
        plt.title(title)
    
    def get_enchanced_image(self,imagepath):
        hr_image = self.preprocess_image(imagepath)
        fake_image = self.model(hr_image)
        fake_image = tf.squeeze(fake_image)
        image_np = fake_image.numpy()
        image_np = np.clip(image_np, 0, 255)
        image_pil = Image.fromarray(image_np.astype(np.uint8))
        cv_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        return cv_image