import cv2
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

class Image_board_detection:
    """Procces img"""
    def __init__(self):
        pass
    def img_read(self, img_dir="img"):  
        img_data = []
        cwd = os.getcwd()
        full_path = os.path.join(cwd, img_dir)
        if not os.path.isdir(full_path):
            exit (f"Directory '{full_path}' doesn't exist.")
        images = [file for file in os.listdir(full_path) if file.endswith(('.png', '.jpg', '.jpeg'))]
        img_data = [cv2.imread(os.path.join(full_path, img)) for img in images]
        return img_data
    
    def img_color_transform(self):
        contours_size = {}
        img_data = self.img_read(img_dir="img/type1")
        min_contour_area = 2500
        max_contour_area = 4500
        for i, img in enumerate(img_data):
            if img is None:
                continue
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_bright = cv2.convertScaleAbs(img_gray, alpha=4, beta=0)
            img_blurred = cv2.GaussianBlur(img_bright , (9, 9), 0)            
            _, thresholding = cv2.threshold(img_blurred, 110, 255, cv2.THRESH_BINARY)     
            contours, _ = cv2.findContours(thresholding, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_size[i] = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area and cv2.contourArea(cnt) < max_contour_area]
        #-----------------------------------------------------------------------------------
        #Do usunięcia - do podglądu konturów
            img_with_contours = img.copy()
            cv2.drawContours(img_with_contours, contours_size[i], -1, (0, 255, 0), 2)
            img_resized = cv2.resize(img_with_contours , None, fx=0.35, fy=0.35)
            cv2.imshow(str(i), img_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #-----------------------------------------------------------------------------------
        return contours_size
        
    def contours_area_and_perimeter(self):
        contours_size = self.img_color_transform()
        areas = []
        perimeters = []
        for img_index, contours in contours_size.items():
            for contour in contours:           
                area = cv2.contourArea(contour) 
                perimeter = cv2.arcLength(contour, True)
                areas.append(area)
                perimeters.append(perimeter)                
                #print(f"pole:{area}, obwód{perimeter}")
        return areas, perimeters
    
    def position_calc(self):
        contours_size = self.img_color_transform()  
        img_list = {}
        for img_index, contours in contours_size.items():
            positions = {}
            for i, contour in enumerate(contours):
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    pos = []  
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    pos.append(cx)
                    pos.append(cy)
                else:
                    pos.append(None)     
                positions[i] = pos          
            img_list[img_index] = positions
        return img_list
    
    def position_match(self):
        img_list = self.position_calc()
        for img_k, img_refer in img_list[0].items():
            for img_key, img_comp in img_list.items():
                if img_key != 0:
                    for pos in img_comp.values():
                        print(img_refer)
                        print(pos)
                        x1 = 9999
                        x2 = np.sqrt((pos[0] - img_refer[0])**2 + (pos[1] - img_refer[1])**2)
                        print(x2)
                        print('------------------------------------------')
                        
    def plot_normal_distribution(self, data, title, xlabel):
        plt.hist(data, bins=10, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        
        mean = np.mean(data)
        std_dev = np.std(data)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mean, std_dev)
        plt.plot(x, p, 'r', linewidth=2, label=f"Rozkład normalny\n$\\mu={mean:.2f}, \\sigma={std_dev:.2f}$")

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Gęstość")
        plt.legend()
        plt.show()

reader = Image_board_detection()
areas, perimeters = reader.contours_area_and_perimeter()
reader.position_calc()
reader.position_match()
#reader.plot_normal_distribution(areas, "Rozkład normalny dla pól konturów", "Pole")
#reader.plot_normal_distribution(perimeters, "Rozkład normalny dla obwodów konturów", "Obwód")