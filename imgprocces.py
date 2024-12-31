import cv2
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import seaborn as sns

class Image_board_detection:
    """Procces img"""
    def __init__(self, img_dir: str = "img"):
        """
        Initialize the class with the directory containing images.
        Parameters:
        img_dir (str): Relative or absolute path to the image directory.
        """
        self.img_dir = img_dir  # Dynamically set the directory


    def img_read(self) -> list[np.ndarray]:
        """
        Reads all images in the specified directory and returns them as numpy arrays.
        
        Returns:
        list[np.ndarray]: List of loaded images.
        """
        img_data: list[np.ndarray] = []
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Directory '{self.img_dir}' does not exist.")
        
        images: list[str] = [file for file in os.listdir(self.img_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found images: {images}")  # Debugging line
        
        for img in images:
            img_path = os.path.join(self.img_dir, img)
            image = cv2.imread(img_path)
            if image is not None:
                img_data.append(image)
            else:
                print(f"Warning: Failed to load image {img_path}")
        return img_data
    
    def img_color_transform(self, min_contour_area: int = 2500, max_contour_area: int = 4500) -> dict[int, list[np.ndarray]]:
        """
        Detects contours in grayscale-transformed images.
        
        Returns:
        dict[int, list[np.ndarray]]: Dictionary mapping image indices to their contours.
        """
        contours_size: dict[int, list[np.ndarray]] = {}
        img_data: list[np.ndarray] = self.img_read()
        
        for i, img in enumerate(img_data):
            if img is None:
                continue
            img_gray: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_bright: np.ndarray = cv2.convertScaleAbs(img_gray, alpha=4, beta=0)
            img_blurred: np.ndarray = cv2.GaussianBlur(img_bright, (9, 9), 0)
            _, thresholding = cv2.threshold(img_blurred, 110, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresholding, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size
            contours_size[i] = [cnt for cnt in contours if min_contour_area < cv2.contourArea(cnt) < max_contour_area]

            # Debug visualization
            img_with_contours = img.copy()
            cv2.drawContours(img_with_contours, contours_size[i], -1, (0, 255, 0), 2)
            img_resized = cv2.resize(img_with_contours, None, fx=0.35, fy=0.35)
            cv2.imshow(str(i), img_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return contours_size
        
    def contours_area_and_perimeter(self) -> tuple[list[float], list[float]]:
        contours_size: dict[int, list[np.ndarray]] = self.img_color_transform()
        areas: list[float] = []
        perimeters: list[float] = []
        for img_index, contours in contours_size.items():
            for contour in contours:
                area: float = cv2.contourArea(contour)
                perimeter: float = cv2.arcLength(contour, True)
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

    def compare_areas_from_lists(self, areas1: list[float], areas2: list[float]) -> list[float]:
        if len(areas1) != len(areas2):
            raise ValueError(f"Arrays of varying sizes: {len(areas1)} , {len(areas2)} cannot be compared")

        percentage_differences: list[float] = []
        for i, (area1, area2) in enumerate(zip(areas1, areas2)):
            if area2 == 0:
                diff: float = float('inf')  # division by zero safeguard
            else:
                diff = abs(area1 - area2) / area2 * 100
            percentage_differences.append(diff)
            print(f"Index {i}: Area1 = {area1:.2f}, Area2 = {area2:.2f}, % Difference = {diff:.2f}%")
        return percentage_differences

    def create_mask(self, contours: list[np.ndarray], img_shape: tuple[int, int, int]) -> np.ndarray:
        """Creates a mask for given contours."""
        mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
        for contour in contours:
            cv2.drawContours(mask, [contour], -1, color=255, thickness=cv2.FILLED)
        return mask

    def create_individual_masks(self, contours: list[np.ndarray], img_shape: tuple[int, int]) -> list[np.ndarray]:
        """creates individual masks for each grain"""
        individual_masks = []
        for contour in contours:
            mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, color=255, thickness=cv2.FILLED)
            individual_masks.append(mask)
        return individual_masks

    def extract_pixels_with_mask(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """extracts pixels from an image using a given mask."""
        return cv2.bitwise_and(img, img, mask=mask)
    
    def analyze_grains_sequentially(self, img_indices: list[int], diff_threshold: int = 10):
        """
        Analyzes grains across images sequentially.
        
        Parameters:
        img_indices (list[int]): List of image indices to analyze.
        diff_threshold (int): Threshold for pixel differences.
        
        Returns:
        dict: Results of grain analysis between images.
        """
        contours_size = self.img_color_transform()
        img_data = self.img_read()

        if len(img_indices) < 2:
            raise ValueError("At least two images are required for sequential comparison.")

        grain_analysis_results = {}

        # Analyze each grain sequentially
        for i in range(len(img_indices) - 1):
            img_index1 = img_indices[i]
            img_index2 = img_indices[i + 1]

            if img_index1 not in contours_size or img_index2 not in contours_size:
                raise ValueError(f"Contours not found for one or both images: {img_index1}, {img_index2}.")

            contours1 = contours_size[img_index1]
            contours2 = contours_size[img_index2]

            # Handle mismatch in contour counts gracefully
            num_contours = min(len(contours1), len(contours2))

            grain_comparisons = []
            for grain_idx in range(num_contours):
                # Create masks
                mask1 = self.create_mask([contours1[grain_idx]], img_data[img_index1].shape)
                mask2 = self.create_mask([contours2[grain_idx]], img_data[img_index2].shape)

                # Extract pixels
                extracted_pixels1 = self.extract_pixels_with_mask(img_data[img_index1], mask1)
                extracted_pixels2 = self.extract_pixels_with_mask(img_data[img_index2], mask2)

                # Ensure sizes match
                min_len = min(len(extracted_pixels1[mask1 == 255]), len(extracted_pixels2[mask2 == 255]))
                diff = np.abs(
                    extracted_pixels1[mask1 == 255][:min_len] -
                    extracted_pixels2[mask2 == 255][:min_len]
                )

                # Compute statistics
                mean_diff = np.mean(diff)
                std_diff = np.std(diff)

                grain_comparisons.append({
                    "grain_idx": grain_idx,
                    "mean_diff": mean_diff,
                    "std_diff": std_diff,
                    "image1_index": img_index1,
                    "image2_index": img_index2,
                })

            grain_analysis_results[f"{img_index1}-{img_index2}"] = grain_comparisons

        return grain_analysis_results
    

    def visualize_results(self, grain_analysis_results):
        """
        Visualizes the sequential grain comparison results.

        Parameters:
        grain_analysis_results (dict): Results of grain analysis to visualize.
        """
        # Loop through all comparisons
        for comparison_key, grain_comparisons in grain_analysis_results.items():
            print(f"Comparison: {comparison_key}")  # Print comparison details
            fig, ax = plt.subplots(figsize=(10, 6))  # Create a plot for each comparison
            
            grain_indices = []
            mean_diffs = []
            std_diffs = []

            # Collect data for plotting
            for grain in grain_comparisons:
                grain_indices.append(grain['grain_idx'])
                mean_diffs.append(grain['mean_diff'])
                std_diffs.append(grain['std_diff'])

                # Print details for debugging
                print(f"Grain {grain['grain_idx']}: Mean Diff = {grain['mean_diff']:.2f}, Std = {grain['std_diff']:.2f}")

            # Plot mean differences with error bars
            ax.errorbar(grain_indices, mean_diffs, yerr=std_diffs, fmt='-o', label=f"Comparison {comparison_key}")

            ax.set_title(f"Grain Comparison for {comparison_key}")
            ax.set_xlabel("Grain Index")
            ax.set_ylabel("Mean Pixel Difference")
            ax.legend()
            ax.grid(True)

            # Show the plot
            plt.tight_layout()
            plt.show()
    

    def compare_pixels(self, img_index1: int, img_index2: int):
        contours_size = self.img_color_transform()
        img_data = self.img_read(img_dir="img/type1")

        if img_index1 not in contours_size or img_index2 not in contours_size:
            raise ValueError(f"Contours not found for one or both images: {img_index1}, {img_index2}.")
        img1 = img_data[img_index1]
        img2 = img_data[img_index2]
        contours = contours_size[img_index1]

        # create a mask
        mask = self.create_mask(contours, img1.shape)

        # extract pixels
        extracted_img1 = self.extract_pixels_with_mask(img1, mask)
        extracted_img2 = self.extract_pixels_with_mask(img2, mask)

        return extracted_img1, extracted_img2, mask

    def compare_pixels_between_images(self, img_index1=0, img_index2=1):
        extracted_img1, extracted_img2, mask = self.compare_pixels(img_index1, img_index2)

        # compute pixel differences
        diff = cv2.absdiff(extracted_img1, extracted_img2)

        # compute statistics
        mean_diff = np.mean(diff[mask == 255])
        std_diff = np.std(diff[mask == 255])

        print(f"Mean Pixel Difference: {mean_diff}")
        print(f"Standard Deviation of Pixel Difference: {std_diff}")

        return {"mean_difference": mean_diff, "std_difference": std_diff}

    def compare_pixels_with_heatmap(self, img_index1=0, img_index2=1):
        extracted_img1, extracted_img2, mask = self.compare_pixels(img_index1, img_index2)

        # compute pixel differences
        diff = cv2.absdiff(extracted_img1, extracted_img2)

        # generate heatmap data
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        heatmap_data = np.zeros_like(gray_diff, dtype=np.float32)
        heatmap_data[mask == 255] = gray_diff[mask == 255]

        # normalize heatmap data for visualization
        heatmap_normalized = heatmap_data / np.max(heatmap_data)

        # display heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_normalized, cmap='jet', cbar=True, square=True)
        plt.title("Heatmap of Pixel Differences")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show()

    def extract_pixels_in_contours(self, img_index: int = 0) -> np.ndarray:
        contours_size = self.img_color_transform()
        img_data = self.img_read(img_dir="img/type1")

        if img_index not in contours_size:
            raise ValueError(f"No contours found for image index {img_index}.")
        img = img_data[img_index]
        contours = contours_size[img_index]

        # creates mask and extracts pixels
        mask = self.create_mask(contours, img.shape)
        return self.extract_pixels_with_mask(img, mask)


    def plot_normal_distribution(self, data: list[float], title: str, xlabel: str) -> None:
        plt.hist(data, bins=10, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        
        mean: float = np.mean(data)
        std_dev: float = np.std(data)
        xmin, xmax = plt.xlim()
        x: np.ndarray = np.linspace(xmin, xmax, 100)
        p: np.ndarray = norm.pdf(x, mean, std_dev)
        plt.plot(x, p, 'r', linewidth=2, label=f"Rozkład normalny\n$\\mu={mean:.2f}, \\sigma={std_dev:.2f}$")

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Gęstość")
        plt.legend()
        plt.show()

reader = Image_board_detection("img\\type2")
#areas, perimeters = reader.contours_area_and_perimeter()

#areas_image1 = [3100.0, 2750.0, 3300.0]
#areas_image2 = [3000.0, 2700.0, 3200.0]
#percentage_differences = reader.compare_areas_from_lists(areas_image1, areas_image2)

#reader.position_calc()
#reader.position_match()
#reader.plot_normal_distribution(areas, "Rozkład normalny dla pól konturów", "Pole")
#reader.plot_normal_distribution(perimeters, "Rozkład normalny dla obwodów konturów", "Obwód")

""" Specify image numbers to analyze """
image_indices = [0,1,2]
#grain_results = reader.analyze_grains_across_images(image_indices, ref_img_index=0)


grain_results = reader.analyze_grains_sequentially(image_indices)
reader.visualize_results(grain_results)
#print("Extracting and displaying pixels in contours...")
#reader.extract_pixels_in_contours(img_index=1)
#comparison_result = reader.compare_pixels_between_images(img_index1=0, img_index2=1)
#reader.compare_pixels_with_heatmap(img_index1=0, img_index2=1)