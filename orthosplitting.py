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
image = np.ones((150, 250), dtype=rasterio.ubyte) * 127

with rasterio.open(ortho_file) as src:
    ortho = src.read()
    ortho_meta = src.meta
    columns = src.width
    rows = src.height
    bands = src.count   
    # print(ortho_meta) # print metadata
    print(ortho.shape)
    print(columns, rows, bands)
    # Define the tile coordinates (upper left corner and lower right corner)
    tile_ulc = (2000, 2000)  # Upper left corner (row, col)
    tile_lrc = (2500, 2500)  # Lower right corner (row, col)
    
    # Create a window using the tile coordinates
    # window_location = Window.from_slices((tile_ulc[0], tile_lrc[0]), (tile_ulc[1], tile_lrc[1]))    
    window_location = Window.from_slices(slice(10, -10), slice(10, -10), height=1000, width=1000)
    
    img = src.read(window=window_location)
    print(img.shape)
    # cv.imwrite(ortho_path + "/test.png", img)
    temp = img.transpose(1, 2, 0)
    t2 = cv.split(temp)
    img_cv = cv.merge([t2[2], t2[1], t2[0]])
    cv.imshow("test", img_cv)
    cv.waitKey(0)
                                  
                      
                   


# with rasterio.open(   output_file, 'w',
#         driver='GTiff', width=500, height=300, count=1,
#         dtype=image.dtype) as dst:
#     dst.write(img, window=Window(50, 30, 250, 150), indexes=1)