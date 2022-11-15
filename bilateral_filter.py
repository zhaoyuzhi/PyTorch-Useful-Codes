import cv2
import math
import numpy as np

if __name__ == "__main__":
    
    """
    image_matrix: the image matrix, its type is ndarray
    window_length : the diamter of filter
    sigma_color: the sigma for the gaussion weight of color
    sigma_space: the sigma for the gaussion weight of weight
    mask_image_matrix: the image matrix mask,where if mask_image_matrix[i][j]==0,it means filtering the pixel in (i,j).default=None
    """

    def bilateral_filter_image(image_matrix, window_length=7, sigma_color=25, sigma_space=9, mask_image_matrix = None):
        mask_image_matrix = np.zeros(
            (image_matrix.shape[0], image_matrix.shape[1])) if mask_image_matrix is None else mask_image_matrix #default: filtering the entire image
        image_matrix = image_matrix.astype(np.int32) #transfer the image_matrix to type int32，for uint cann't represent the negative number afterward
    
        def limit(x):
            x = 0 if x < 0 else x
            x = 255 if x > 255 else x
            return x
        limit_ufun = np.vectorize(limit, otypes=[np.uint8])
        def look_for_gaussion_table(delta):
            return delta_gaussion_dict[delta]
        def generate_bilateral_filter_distance_matrix(window_length,sigma):
            distance_matrix = np.zeros((window_length,window_length,3))
            left_bias = int(math.floor(-(window_length - 1) / 2))
            right_bias = int(math.floor((window_length - 1) / 2))
            for i in range(left_bias,right_bias+1):
                for j in range(left_bias,right_bias+1):
                    distance_matrix[i-left_bias][j-left_bias] = math.exp(-(i**2+j**2)/(2*(sigma**2)))
            return distance_matrix
        delta_gaussion_dict = {i: math.exp(-i ** 2 / (2 *(sigma_color**2))) for i in range(256)}
        look_for_gaussion_table_ufun = np.vectorize(look_for_gaussion_table, otypes=[np.float64])#to accelerate the process of get the gaussion matrix about color.key:color difference，value:gaussion weight
        bilateral_filter_distance_matrix = generate_bilateral_filter_distance_matrix(window_length,sigma_space)#get the gaussion weight about distance directly

        margin = int(window_length / 2)
        left_bias = math.floor(-(window_length - 1) / 2)
        right_bias = math.floor((window_length - 1) / 2)
        filter_image_matrix = image_matrix.astype(np.float64)

        for i in range(0 + margin, image_matrix.shape[0] - margin):
            for j in range(0 + margin, image_matrix.shape[1] - margin):
                if mask_image_matrix[i][j]==0:
                    filter_input = image_matrix[i + left_bias:i + right_bias + 1,
                                j + left_bias:j + right_bias + 1] #get the input window
                    bilateral_filter_value_matrix = look_for_gaussion_table_ufun(np.abs(filter_input-image_matrix[i][j])) #get the gaussion weight about color
                    bilateral_filter_matrix = np.multiply(bilateral_filter_value_matrix, bilateral_filter_distance_matrix) #multiply color gaussion weight  by distane gaussion weight to get the no-norm weigth matrix
                    bilateral_filter_matrix = bilateral_filter_matrix/np.sum(bilateral_filter_matrix,keepdims=False,axis=(0,1)) #normalize the weigth matrix
                    filter_output = np.sum(np.multiply(bilateral_filter_matrix,filter_input),axis=(0,1)) #multiply the input window by the weigth matrix，then get the sum of channels seperately
                    filter_image_matrix[i][j] = filter_output
        filter_image_matrix = limit_ufun(filter_image_matrix) #limit the range
        return filter_image_matrix

    image = cv2.imread('F:\\dataset\\COCO2014_processed\\COCO2014_test_256\\COCO_test2014_000000000001.jpg')
    cv2.imwrite("image.png", image)
    filtered_image_OpenCV = cv2.bilateralFilter(image, 7, 20.0, 20.0)
    cv2.imwrite("filtered_image_OpenCV.png", filtered_image_OpenCV)
    bilateral_filtered = bilateral_filter_image(image, 7, 20.0, 20.0)
    cv2.imwrite("filtered_image_own.png", bilateral_filtered)
