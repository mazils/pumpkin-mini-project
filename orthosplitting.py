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

class OrthoSplitter:
    def __init__(self, ortho_file, output_file):
        self.ortho_file = ortho_file
        self.output_file = output_file
        self.ortho = None
        self.ortho_meta = None
        self.columns = None
        self.rows = None
        self.bands = None
        self.window_location = None
        self.img = None
        self.ortho_meta = None

    def load_ortho(self):
        with rasterio.open(self.ortho_file) as src:
            self.ortho = src.read()
            self.ortho_meta = src.meta
            self.columns = src.width
            self.rows = src.height
            self.bands = src.count   
            # print(ortho_meta) # print metadata
            print(self.ortho.shape)
            print(self.columns, self.rows, self.bands)
            # Define the tile coordinates (upper left corner and lower right corner)
            tile_ulc = (2310, 1474)  # Upper left corner (row, col)
            tile_lrc = (5813, 11487)  # Lower right corner (row, col)
            
            # Create a window using the tile coordinates
            # window_location = Window.from_slices((tile_ulc[0], tile_lrc[0]), (tile_ulc[1], tile_lrc[1]))    
            self.window_location = Window.from_slices(slice(1, -1), slice(1, -1), height=5000, width=10000)
            
            self.img = src.read(window=self.window_location)
            print(self.img.shape)
            # Update metadata to reflect the new dimensions
            self.ortho_meta.update({
                "height": self.window_location.height,
                "width": self.window_location.width,
                "transform": rasterio.windows.transform(self.window_location, src.transform)
            })

            # Write the image to the output directory
            with rasterio.open(self.output_file, 'w', **self.ortho_meta) as dst:
                dst.write(self.img)




def main():
    ortho_splitter = OrthoSplitter(ortho_file, output_file)
    ortho_splitter.load_ortho()
    # with rasterio.open(ortho_file) as src:
    #     ortho = src.read()
    #     ortho_meta = src.meta
    #     columns = src.width

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
    tile_ulc = (2310, 1474)  # Upper left corner (row, col)
    tile_lrc = (5813, 11487)  # Lower right corner (row, col)
    
    # Create a window using the tile coordinates
    # window_location = Window.from_slices((tile_ulc[0], tile_lrc[0]), (tile_ulc[1], tile_lrc[1]))    
    window_location = Window.from_slices(slice(1, -1), slice(1, -1), height=5000, width=10000)
    
    img = src.read(window=window_location)
    print(img.shape)
    # Update metadata to reflect the new dimensions
    ortho_meta.update({
        "height": window_location.height,
        "width": window_location.width,
        "transform": rasterio.windows.transform(window_location, src.transform)
    })

    # Write the image to the output directory
    with rasterio.open(output_file, 'w', **ortho_meta) as dst:
        dst.write(img)
    # temp = img.transpose(1, 2, 0)
    # t2 = cv.split(temp)
    # img_cv = cv.merge([t2[2], t2[1], t2[0]])
    # cv.imshow("test", img_cv)
    # cv.waitKey(0)
                                  
                      
                   


# with rasterio.open(   output_file, 'w',
#         driver='GTiff', width=500, height=300, count=1,
#         dtype=image.dtype) as dst:
#     dst.write(img, window=Window(50, 30, 250, 150), indexes=1)