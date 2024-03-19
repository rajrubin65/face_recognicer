import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import array_to_img,load_img, img_to_array



def plot_image(org_image,processde_image,title='Pre processed'):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.axis('off')
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(org_image)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title('Processd Image')
    plt.imshow(processde_image)
    plt.axis('off')
    plt.show()


def apply_gausian_blur(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blurred_image = cv2.GaussianBlur(image_rgb, (5, 5), 0)
    plot_image(image,blurred_image,'Gausian Blur')
    return blurred_image


def apply_normalization(image_path):
    """
    Normalizing the images by resizing them to a fixed size helps to mitigate 
    variations in facial scale.
    """
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Normalize the image
    normalized_image = cv2.normalize(gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    plot_image(image,normalized_image,'Normalization')
    return normalized_image



def apply_data_augmentation(image_path, num_of_image=1):
    def rotate_image(image, angle):
        rows, cols, _ = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
        return rotated_image

    def flip_image(image, direction):
        flipped_image = cv2.flip(image, direction)
        return flipped_image

    def apply_blur(image):
        kernel_size = np.random.choice([3, 5, 7])
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred_image
    try:
        # Load an example image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Generate augmented images
        images = []
        for _ in range(num_of_image):
            # Randomly select an augmentation method
            aug_method = np.random.choice(['rotate'])

            if aug_method == 'rotate':
                # Randomly rotate the image
                angle = np.random.randint(-30, 30)
                img_aug = rotate_image(img, angle)
            elif aug_method == 'flip':
                # Randomly flip the image horizontally or vertically
                flip_direction = np.random.choice([-1, 0, 1])
                img_aug = flip_image(img, flip_direction)
            elif aug_method == 'blur':
                # Apply random Gaussian blur
                img_aug = apply_blur(img)

            images.append(img_aug)
        # plot_image(*images)
        return images
    except Exception as e:
        print("Error:", e)
        return None


# Function to enhance image sharpness, contrast and apply Gaussian blur
def enhance_image(image_path,sharpness=4, contrast=1.3, blur=3,check_sharp_cont=True):
    """Enhance image sharpness, contrast, and blur.

    Args:
        image_path (str): Path to the input image.
        sharpness (float, optional): Sharpness level. Defaults to 4.
        contrast (float, optional): Contrast level. Defaults to 1.3.
        blur (int, optional): Blur level. Defaults to 3.
    """
    
    # Load the image
    # img = cv2.imread(image_path)
    img = image_path
    # Convert the image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    is_not_sharp, is_not_contr = assess_image_quality(img)
    if any((is_not_sharp,is_not_contr)) and check_sharp_cont:
        # Convert the image to PIL Image
        pil_img = Image.fromarray(img)
        # Enhance the sharpness
        enhancer = ImageEnhance.Sharpness(pil_img)
        img_enhanced = enhancer.enhance(sharpness)
        # Enhance the contrast
        enhancer = ImageEnhance.Contrast(img_enhanced)
        img_enhanced = enhancer.enhance(contrast)
        # Convert back to OpenCV image (numpy array)
        img_enhanced = np.array(img_enhanced)
        # Apply a small amount of Gaussian blur
        img_enhanced = cv2.GaussianBlur(img_enhanced, (blur, blur), 0)
        # plot_image(img,img_enhanced)
        return img_enhanced
    # plot_image(img,img,'not processed')
    return img



def assess_image_quality(img, sharpness_threshold=800, contrast_threshold=70):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculate the Laplacian variance for sharpness
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Calculate the standard deviation of pixel intensities for contrast
    contrast_var = gray.std()
    # Assess sharpness and contrast based on thresholds
    sharpness = laplacian_var <= sharpness_threshold
    contrast = contrast_var <= contrast_threshold
    print('saturation>',laplacian_var,'contrast>',contrast_var)
    return sharpness, contrast




def blur_background(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply Gaussian blur to the grayscale image
    blurred_gray = cv2.GaussianBlur(gray, (51, 51), 0)
    # Cnvert the blurred grayscale image to BGR
    blurred_BGR = cv2.cvtColor(blurred_gray, cv2.COLOR_GRAY2RGB)
    # Create a mask by thresholding the blurred image
    _, mask = cv2.threshold(blurred_gray, 200, 255, cv2.THRESH_BINARY)
    # Invert the mask
    mask_inv = cv2.bitwise_not(mask)
    # Combine the original image with the blurred background using the mask
    blurred_background = cv2.bitwise_and(image, image, mask=mask_inv) + blurred_BGR
    plot_image(image,blurred_background,'Blur Background')
    return blurred_background