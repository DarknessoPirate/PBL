import cv2
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm

class Image_board_detection:
    """Procces img"""
    def __init__(self):
        pass
    
    @staticmethod
    def contours_area_and_perimeter(contour):     
                area = cv2.contourArea(contour) 
                perimeter = cv2.arcLength(contour, True)            
                print(f"area:{area}, perimeter: {perimeter}")

    @staticmethod
    def _calculate_middle(contour) -> list:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            pos = [cx, cy]
        else:
            pos = [None, None]
        return pos
    
    @staticmethod
    def _board_points(contour) -> list:
        leftmost = tuple(contour[contour[:, 0].argmin()])
        rightmost = tuple(contour[contour[:, 0].argmax()])
        topmost = tuple(contour[contour[:, 1].argmin()])
        bottommost = tuple(contour[contour[:, 1].argmax()])
        points = [leftmost, rightmost, topmost, bottommost]
        return points

    @staticmethod
    def _linear_regr(point1, point2) -> tuple:
        x_coords = np.array([point1[0], point2[0]])
        y_coords = np.array([point1[1], point2[1]])
        coefficients = np.polyfit(x_coords, y_coords, 1)
        slope, intercept = coefficients  
        coef = (slope, intercept)
        return coef

    @staticmethod
    def img_read(img_dir="img") -> list:  
        cwd = os.getcwd()
        full_path = os.path.join(cwd, img_dir)
        if not os.path.isdir(full_path):
            exit (f"Directory '{full_path}' doesn't exist.")
        images = [file for file in os.listdir(full_path) if file.endswith(('.png', '.jpg', '.jpeg', 'JPEG'))]
        img_data = [cv2.imread(os.path.join(full_path, img)) for img in images]
        return img_data
    
    def img_color_transform(self, min_contour_area = 2500, max_contour_area = 4500) -> list:
        contours_size = {}
        img_data = self.img_read(img_dir="img\masks") #<---------Tutaj na razie wpisuje się lokalizacje ze zdjęciami
        for i, img in enumerate(img_data):
            if img is None: continue          
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_bright = cv2.convertScaleAbs(img_gray, alpha=4, beta=6)
            img_blurred = cv2.GaussianBlur(img_bright , (9, 9), 0)
            threshold_value = np.mean(img_blurred) + np.std(img_blurred)          
            _, thresholding = cv2.threshold(img_blurred, threshold_value, 255, cv2.THRESH_BINARY)     
            contours, _ = cv2.findContours(thresholding, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours_size[i] = [cnt for cnt in contours if cv2.contourArea(cnt) > 
                                min_contour_area and cv2.contourArea(cnt) < max_contour_area]
        #-----------------------------------------------------------------------------------
        #Do usuniÄcia - do podglÄdu konturĂłw
            img_with_contours = img.copy()
            cv2.drawContours(img_with_contours, contours_size[i], -1, (0, 255, 0), 2)
            img_resized = cv2.resize(thresholding , None, fx=0.35, fy=0.35)
            cv2.imshow(str(i), img_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #-----------------------------------------------------------------------------------
        return contours_size
    
    def position_calc(self) -> dict:
        contours_size = self.img_color_transform()  
        img_dict = {}
        for img_index, contours in contours_size.items():
            positions = {}
            for i, contour in enumerate(contours):
                contour_data = {}
                contour_data["contour"] = contour
                pos = self._calculate_middle(contour)
                contour_data["mid"] = pos  
                positions[i] = contour_data          
            img_dict[img_index] = positions
        return img_dict
    
    def position_match(self) -> dict:
        img_dict = self.position_calc()
        data = {}
        for img_k, img_refer in img_dict[0].items():
            for img_key, img_comp in img_dict.items():
                if img_key != 0:
                    x1 = np.inf
                    y = [0,0]
                    contour_dict = {"contour_ref": img_refer["contour"]}
                    for pos in img_comp.values():
                        x2 = math.dist(img_refer["mid"], pos["mid"])
                        if x1 > x2:
                            x1 = x2
                            y = [pos["mid"][0] - img_refer["mid"][0] , pos["mid"][1] - img_refer["mid"][1]]
                            contour_dict[f"contour"] = pos["contour"]
                            contour_dict[f"diff"] = y
                    if img_key not in data:
                        data[img_key] = {}  
                    data[img_key][img_k] = contour_dict
        return data
    
    def match_shape(self) -> dict:
        data = self.position_match()
        for png_nr, png in data.items():
            dnorm = []
            for cnt_dict in png.values():
                cmp = cv2.matchShapes(cnt_dict['contour_ref'],cnt_dict['contour'],cv2.CONTOURS_MATCH_I1,0.0)
                dnorm.append(cmp)
            mean = np.mean(dnorm)
            std_dev = np.std(dnorm)
            after_filter = [keys for keys, cnt_dict in png.items() 
                            if (cmp := cv2.matchShapes(cnt_dict['contour_ref'], 
                                                       cnt_dict['contour'], cv2.CONTOURS_MATCH_I1, 0.0)) > mean + std_dev]
            for keys_to_remove in after_filter:
                del data[png_nr][keys_to_remove]
        return data

    def img_compare(self)-> None:
        data = self.match_shape()
        #-------------------------------------------------------------------
        #Tutaj można odpalić podgląd konturu wybranego
        contour_ref = data[2][10]['contour_ref'].reshape(-1, 2)
        contour_1 = data[2][10]['contour'].reshape(-1, 2)
        difference = data[2][10]['diff']
        #-------------------------------------------------------------------
        self.contours_area_and_perimeter(contour_ref)
        self.contours_area_and_perimeter(contour_1)
        pos0 = self._calculate_middle(contour_1)
        pos1 = self._calculate_middle(contour_ref)
        points0 = self._board_points(contour_1)
        points1 = self._board_points(contour_ref)

        contour_1_diff = contour_1 - difference
  
        pos2 = self._calculate_middle(contour_1_diff)

        pts1 = []
        for pts in points1:
            board_line = {}
            coef = self._linear_regr(pos1, pts)
            board_line["point"] = pts
            board_line["coef"] = coef 
            pts1.append(board_line)   
        
        points_diff = []
        peri = cv2.arcLength(contour_1_diff, True)
        r = peri/(2 * np.pi)
 
        for ptt in pts1:
            y_val_ = np.inf
            y_val1_ = np.inf
            for cnt_pt in contour_1_diff:
                y_vals = ptt["coef"][1] + ptt["coef"][0] * cnt_pt[0]
                y_comp = abs(cnt_pt[1]-y_vals) 
                if y_comp < y_val_:
                    y_val_ = y_comp
                    cnt_pt1 = cnt_pt

            for cnt_pt in contour_1_diff:
                y_vals = ptt["coef"][1] + ptt["coef"][0] * cnt_pt[0]
                y_comp = abs(cnt_pt[1]-y_vals)
                if y_comp < y_val1_ and y_comp != y_val_:
                    y_val1_ = y_comp
                    line_lengh = math.dist(cnt_pt, cnt_pt1)
                    if line_lengh > r:
                        cnt_pt2 = cnt_pt 
            cnt_pt1_lengh = math.dist(ptt["point"], cnt_pt1)
            cnt_pt2_lengh = math.dist(ptt["point"], cnt_pt2)
            if cnt_pt1_lengh <  cnt_pt2_lengh: points_diff.append(cnt_pt1)
            else: points_diff.append(cnt_pt2)
 

        plt.figure(figsize=(10, 6))

        plt.plot(contour_ref[:, 0], contour_ref[:, 1], label='contour_ref', color='blue')
        plt.plot(contour_1[:, 0], contour_1[:, 1], label='contour_1', color='green')
        plt.plot(contour_1_diff[:, 0], contour_1_diff[:, 1], label='contour_1_diff', color='orange', linestyle='dashed') 

        plt.scatter(pos0[0], pos0[1], color='green')
        plt.scatter(pos1[0], pos1[1], color='blue')
        plt.scatter(pos2[0], pos2[1], color='red')

        for point in points0:
            plt.scatter(point[0], point[1], color='green')
        for point in points1:
            plt.scatter(point[0], point[1], color='blue')
        for point in points_diff:
            plt.scatter(point[0], point[1], color='red')
        for pt in pts1:
            x_vals = np.array(plt.gca().get_xlim())
            y_vals = pt["coef"][1] + pt["coef"][0] * x_vals
            plt.plot(x_vals, y_vals, '--', label=f'Prosta a ={pt["coef"][0]}, b ={pt["coef"][1]}')

        plt.xlabel('Os X')
        plt.ylabel('Os Y')
        plt.title('Porownanie macierzy Kontur z roznicami')
        plt.legend()
        plt.grid()
        plt.show()

    def params_calc(self) -> None:
        data = self.match_shape()
        for png in data.values():
            points_list = []
            points_dist = []
            for cnt_dict in png.values(): 
                contour = cnt_dict["contour"].reshape(-1, 2)
                contour_ref = cnt_dict["contour_ref"].reshape(-1, 2)
                diff = cnt_dict["diff"]
                pos1 = self._calculate_middle(contour_ref)
                points1 =  self._board_points(contour_ref)
                contour_2 = contour - diff

                pts1 = []
                for pts in points1:
                    board_line = {}
                    coef = self._linear_regr(pos1, pts)
                    board_line["point"] = pts
                    board_line["coef"] = coef 
                    pts1.append(board_line)   

                points_diff = []
                peri = cv2.arcLength(contour_2, True)
                r = peri/(2 * np.pi)
        
                for ptt in pts1:
                    y_val_ = np.inf
                    y_val1_ = np.inf
                    for cnt_pt in contour_2:
                        y_vals = ptt["coef"][1] + ptt["coef"][0] * cnt_pt[0]
                        y_comp = abs(cnt_pt[1]-y_vals) 
                        if y_comp < y_val_:
                            y_val_ = y_comp
                            cnt_pt1 = cnt_pt

                    for cnt_pt in contour_2:
                        y_vals = ptt["coef"][1] + ptt["coef"][0] * cnt_pt[0]
                        y_comp = abs(cnt_pt[1]-y_vals)
                        if y_comp < y_val1_ and y_comp != y_val_:
                            y_val1_ = y_comp
                            line_lengh = math.dist(cnt_pt, cnt_pt1)
                            if line_lengh > r:
                                cnt_pt2 = cnt_pt 
                    cnt_pt1_lengh = math.dist(ptt["point"], cnt_pt1)
                    cnt_pt2_lengh = math.dist(ptt["point"], cnt_pt2)
                    if cnt_pt1_lengh <  cnt_pt2_lengh: points_diff.append(cnt_pt1)
                    else: points_diff.append(cnt_pt2) 
                    points_dist.append(math.dist(ptt["point"], points_diff[-1]))
                points_list.append(points_diff)

            # Tworzenie histogramu
            bins = np.arange(0, 5.1, 0.1)
            plt.hist(points_dist, bins=bins, edgecolor='black')

            # Dodanie tytulu i etykiet
            plt.title('Histogram of Quartz Grain Lengths')
            plt.xlabel('Length')
            plt.ylabel('Frequency')
            plt.show()
            # Wczytaj obraz
            img = cv2.imread('img/type1/14_50_36_Pro.jpg') #<---------Tutaj na razie wpisuje się zdjęcie referencyjne

            for cnt_dict in png.values():
                contour_ref = cnt_dict["contour_ref"]
                cv2.drawContours(img, contour_ref, -1, (0, 255, 0), 2)

            for cnt_dict in png.values():
                contour = cnt_dict["contour"]
                cv2.drawContours(img, contour, -1, (255, 0, 0), 2)

            for cnt_dict in png.values():
                contour = cnt_dict["contour"].reshape(-1, 2)
                contour_ref = cnt_dict["contour_ref"].reshape(-1, 2)
                diff = cnt_dict["diff"]
                contour_2 = contour - diff
                cv2.drawContours(img, [contour_2.astype(np.int32)], -1, (0, 0, 255), 2)

            # Zaznacz punkty
            for point_list in points_list:
                for point in point_list:
                    cv2.circle(img, tuple(point), 3, (0, 0, 255), -1)

            for cnt_dict in png.values(): 
                contour_ref = cnt_dict["contour_ref"].reshape(-1, 2)
                points1 = self._board_points(contour_ref)
                for point in points1:
                    cv2.circle(img, tuple(point), 3, (0, 255, 0), -1)

            img = cv2.resize(img , None, fx=0.5, fy=0.5)
            # Wyswietl obraz
            cv2.imshow('Image with Contours and Points', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
                                
    def plot_normal_distribution(self, data, title, xlabel) -> None:
        plt.hist(data, bins=10, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        mean = np.mean(data)
        std_dev = np.std(data)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mean, std_dev)
        plt.plot(x, p, 'r', linewidth=2, label=f"Rozklad normalny\n$\\mu={mean:.2f}, \\sigma={std_dev:.2f}$")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Gestosc")
        plt.legend()
        plt.show()

    def plot_hist(self):
        data = self.position_match()
        for png_nr, png in data.items():
            ar = []
            ar1 =[]
            for nr_dict, cnt_dict in png.items():
                contour_1 = data[png_nr][nr_dict]['contour'].reshape(-1, 2)
                contour_ref = data[png_nr][nr_dict]['contour_ref'].reshape(-1, 2)
                area = cv2.contourArea(contour_1)
                area1 = cv2.contourArea(contour_ref)
                ar.append(area)
                ar1.append(area1)
            plt.hist(ar, bins=len(ar))
            plt.hist(ar1, bins=len(ar1))
            plt.show()

reader = Image_board_detection()
#reader.img_compare()
reader.params_calc()