import os
import cv2 as cv
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.plot import show


script_dir = os.path.dirname(__file__)

input_rel = "../figures"
ortho_rel = "../visualized-qgis-images"
ortho_test_rel = "../test_orthophotos"
output_rel = "../processed-images"


input_path = os.path.join(script_dir, input_rel)
output_path = os.path.join(script_dir, output_rel)  
ortho_path = os.path.join(script_dir, ortho_rel)

ortho_file = ortho_path + "/Gyldensteensvej-9-19-2017-orthophoto.tif"
ortho_test = ortho_test_rel + "/Hans-Christian-Andersen-Airport-10-24-2018-orthophoto.tif"
output_file = ortho_path + "/output.tif"

img = cv.imread(ortho_file)
print(img.shape)
bw = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    
cv.imwrite(output_file, bw)
# cv.imshow("test", img)
# cv.waitKey(0)