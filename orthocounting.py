import os
import cv2, numpy as np
import rasterio

script_dir = os.path.dirname(__file__)

input_rel = "../figures"
output_rel = "../processed-images"

input_path = os.path.join(script_dir, input_rel)
output_path = os.path.join(script_dir, output_rel)  
