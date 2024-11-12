import os
import cv2
import shutil
import numpy as np

RAW_DIR = './raw'
DATA_DIR = './data'            # Directory for CHUNK_SIZE X CHUNK_SIZE images with OBB labels
DATA_YOLO_DIR = './data-yolo'  # Directory for CHUNK_SIZE X CHUNK_SIZE images with YOLO labels
CHUNK_SIZE = 256

def obb_to_yolo(class_name, points):
    x_coords = points[0::2]
    y_coords = points[1::2]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    center_x = ((min_x + max_x) / 2)
    center_y = ((min_y + max_y) / 2)
    width = (max_x - min_x)
    height = (max_y - min_y)
    
    return f"{class_name} {center_x} {center_y} {width} {height}\n"

if os.path.exists(DATA_DIR):
    shutil.rmtree(DATA_DIR)
if os.path.exists(DATA_YOLO_DIR):
    shutil.rmtree(DATA_YOLO_DIR)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATA_YOLO_DIR, exist_ok=True)

for label_file in os.listdir(RAW_DIR):
    if label_file.endswith('.txt'):
        image_file = label_file.replace('.txt', '.jpg')
        image_path = os.path.join(RAW_DIR, image_file)
        label_path = os.path.join(RAW_DIR, label_file)
        
        image = None
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
        else:
            image_path = image_path.replace('.jpg', '.jpeg')
            if os.path.exists(image_path):
                image = cv2.imread(image_path)

        if image is None:
            continue
        
        img_height, img_width = image.shape[:2]
        
        with open(label_path, 'r') as f:
            labels = f.readlines()
            
        for i in range(0, img_width, CHUNK_SIZE):
            for j in range(0, img_height, CHUNK_SIZE):
                x_end, y_end = min(i + CHUNK_SIZE, img_width), min(j + CHUNK_SIZE, img_height)
                
                chunk = np.zeros((CHUNK_SIZE, CHUNK_SIZE, 3), dtype=image.dtype)
                chunk[0:y_end-j, 0:x_end-i] = image[j:y_end, i:x_end]
                
                chunk_name = f"{image_file.replace('.jpg', '')}_{i}_{j}"
                
                yolo_labels = []
                obb_labels = []
                
                for label in labels:
                    parts = label.strip().split()
                    class_name, points = parts[0], list(map(float, parts[1:]))
                    points = [
                        points[0] * img_width, points[1] * img_height,
                        points[2] * img_width, points[3] * img_height,
                        points[4] * img_width, points[5] * img_height,
                        points[6] * img_width, points[7] * img_height
                    ]
                    
                    x_coords = points[0::2]
                    y_coords = points[1::2]
                    
                    if (i <= min(x_coords) < i + CHUNK_SIZE 
                    and j <= min(y_coords) < j + CHUNK_SIZE
                    and i <= max(x_coords) < i + CHUNK_SIZE 
                    and j <= max(y_coords) < j + CHUNK_SIZE):
                        adjusted_points = [
                            (points[0] - i) / CHUNK_SIZE, (points[1] - j) / CHUNK_SIZE,
                            (points[2] - i) / CHUNK_SIZE, (points[3] - j) / CHUNK_SIZE,
                            (points[4] - i) / CHUNK_SIZE, (points[5] - j) / CHUNK_SIZE,
                            (points[6] - i) / CHUNK_SIZE, (points[7] - j) / CHUNK_SIZE
                        ]
                        obb_label = f"{class_name} " + " ".join(map(str, adjusted_points)) + "\n"
                        obb_labels.append(obb_label)
                        
                        yolo_label = obb_to_yolo(class_name, adjusted_points)
                        yolo_labels.append(yolo_label)
                
                if obb_labels:
                    cv2.imwrite(os.path.join(DATA_DIR, f"{chunk_name}.jpg"), chunk)
                    with open(os.path.join(DATA_DIR, f"{chunk_name}.txt"), 'w') as obb_file:
                        obb_file.writelines(obb_labels)
                
                if yolo_labels:
                    cv2.imwrite(os.path.join(DATA_YOLO_DIR, f"{chunk_name}.jpg"), chunk)
                    with open(os.path.join(DATA_YOLO_DIR, f"{chunk_name}.txt"), 'w') as yolo_file:
                        yolo_file.writelines(yolo_labels)
