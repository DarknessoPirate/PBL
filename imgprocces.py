import cv2
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

class Image_board_detection:
    """Procces img"""
    def __init__(self):
       self.img_data = []

    def img_read(self, img_dir="img"):  
        cwd = os.getcwd()
        full_path = os.path.join(cwd, img_dir)
        if not os.path.isdir(full_path):
            exit (f"Directory '{full_path}' doesn't exist.")
        images = [file for file in os.listdir(full_path) if file.endswith(('.png', '.jpg', '.jpeg'))]
        self.img_data = [cv2.imread(os.path.join(full_path, img)) for img in images]
        return self.img_data
    
    def img_color_transform(self):
        self.img_read()
        for i, img in enumerate(self.img_data):
            if img is None:
                print(i)
                continue
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
            _, thresholding = cv2.threshold(img_blurred, 40, 100, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresholding, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            img_with_contours = img.copy()
            cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)
            
            img_resized = cv2.resize(img_with_contours, None, fx=0.5, fy=0.5)
            cv2.imshow(str(i), img_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
reader = Image_board_detection()
reader.img_color_transform()