import os
import cv2
from matplotlib import pyplot as plt

def generate_images_size(path_images):
    print ('Loading images...')
    archives = os.listdir(path_images)
    
    freq_x = [0] * 10
    freq_y = [0] * 10

    for archive in archives:
        image = cv2.imread(path_images +'/'+ archive, 0)
        height, width = image.shape

        if width <= 10:
            freq_x[0] += 1
        else:    
            for size in range(1,10):
                if width > size * 10 and width <= (size+1) * 10: 
                        freq_x[size] += 1
        
        if height <= 10:
            freq_y[0] += 1
        else:    
            for size in range(1,10):
                if height > size * 10 and height <= (size+1) * 10: 
                        freq_y[size] += 1

    print(freq_x)
    print(freq_y)
        

    #plt.plot(x, y, 'ro')
    #plt.show()
if __name__ == "__main__":
    generate_images_size("digits/data/")
