import cv2
import numpy as np


class Paynt():
    def __init__(self, image_path, save_path=None):
        self.image = cv2.imread(image_path)
        self.save_path = save_path

    def oil_painting(self):
        result = cv2.xphoto.oilPainting(self.image, 7, 1)

        self.__save_or_show(result)

    def watercolor(self, sigma_s=100, sigma_r=0.5):
        # sigma_s controls the size of the neighborhood. Range 1 - 200

        # sigma_r controls the how dissimilar colors within the neighborhood will be averaged.
        # A larger sigma_r results in large regions of constant color. Range 0 - 1
        result = cv2.stylization(self.image, sigma_s=sigma_s, sigma_r=sigma_r)

        self.__save_or_show(result)

    def sketch(self, kernel_size=11):
        # kernel_size should be an odd integer
        # Larger the kernel_size, more blurred the image will be and it will lose its subtle features.

        self.sketch_grayscale(kernel_size)

    def sketch_grayscale(self, kernel_size=11):
        # kernel_size should be an odd integer
        # Larger the kernel_size, more blurred the image will be and it will lose its subtle features.

        # Obtain the grayscale image of the original image
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Create an inverse of the gray image
        gray_inv = cv2.bitwise_not(gray_image)
        # Apply Gaussian blur to the image
        blur_image = cv2.GaussianBlur(
            gray_inv, (kernel_size, kernel_size), sigmaX=0, sigmaY=0)
        # Create an inverse of the blur image
        blur_inv = cv2.bitwise_not(blur_image)
        # Perform bit-wise division between the grayscale image and the inverted-blurred image
        result = cv2.divide(gray_image, blur_inv, scale=255)

        self.__save_or_show(result)

    def color_pop(self, hue_lower_bound=[0, 0, 0], hue_upper_bound=[360, 255, 255]):
        # Convert the BGR image to HSV color space
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        # Obtain the grayscale image of the original image
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Set the bounds for the desired "pop" hue
        lower_bound = np.array(hue_lower_bound)
        upper_bound = np.array(hue_upper_bound)

        # Create a mask using the bounds set
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        # Create an inverse of the mask
        mask_inv = cv2.bitwise_not(mask)
        # Filter only the desired color from the original image using the mask (foreground)
        pop = cv2.bitwise_and(self.image, self.image, mask=mask)
        # Filter the regions containing colors other than desired color from the grayscale image (background)
        background = cv2.bitwise_and(gray_image, gray_image, mask=mask_inv)
        # Convert the one channelled grayscale background to a three channelled image
        background = np.stack((background,)*3, axis=-1)
        # Add the foreground and the background to create the final "pop" image
        result = cv2.add(pop, background)

        self.__save_or_show(result)

    def __save_or_show(self, result):
        if self.save_path is None:
            cv2.imshow('image', result)

            if cv2.waitKey(0):
                cv2.destroyAllWindows()
        else:
            cv2.imwrite(self.save_path, result)
