import os
import cv2 as cv
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.plot import show


script_dir = os.path.dirname(__file__)

input_rel = "../figures"
ortho_rel = "../visualized-qgis-images"
output_rel = "../processed-images"


input_path = os.path.join(script_dir, input_rel)
output_path = os.path.join(script_dir, output_rel)  
ortho_path = os.path.join(script_dir, ortho_rel)

ortho_file = ortho_path + "/Gyldensteensvej-9-19-2017-orthophoto.tif"
output_file = output_path + "/output.tif"

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
        self.field = None
        self.img = None
        self.ortho_meta = None

    def load_ortho(self):
        self.dataset = rasterio.open(self.ortho_file)
        self.ortho = self.dataset.read()
        self.ortho_meta = self.dataset.meta
        self.columns = self.dataset.width
        self.rows = self.dataset.height
        self.bands = self.dataset.count   
        print("Ortho loaded")    
        print(f"Initial Values = Columns: {self.columns}, Rows: {self.rows}, Bands: {self.bands}")
        
        return self.ortho, self.ortho_meta, self.columns, self.rows, self.bands
    
            
    def crop_field(self):
        """
        Crop the orthophoto to the field of interest
        """
        # Define the tile coordinates (upper left corner and lower right corner)
        tile_ulc = (2310, 1000)  # Upper left corner (row, col)
        tile_lrc = (5810, 12000)  # Lower right corner (row, col)
        
        # Create a window using the tile coordinates
        self.window_location =  Window.from_slices((tile_ulc[0], tile_lrc[0]), (tile_ulc[1], tile_lrc[1]))    
        
        self.field = self.dataset.read(window=self.window_location)
        print(self.field.shape)
        # Update metadata to reflect the new dimensions
        self.ortho_meta.update({
            "height": self.window_location.height,
            "width": self.window_location.width,
            "transform": rasterio.windows.transform(self.window_location, self.dataset.transform)
        })
        
        return self.field
    
    
    def split_field(self, field = None, slices = 4):
        """
        Split the field into smaller tiles
        """
        if field is None:   
            field = self.field
        # Current Shape of the field
        bands, rows, cols = field.shape 
        
        # Calculate the number of rows and columns for each tile
        row_slices = rows // slices
        col_slices = cols // slices
        print(row_slices, col_slices)

        for i in range(slices):
            for j in range(slices):
                # Calculate the window coordinates for each tile
                window = Window.from_slices((i * row_slices, (i + 1) * row_slices), (j * col_slices, (j + 1) * col_slices))
                print(window)
                # Read the tile
                tile = field[:, window.row_off:window.row_off + window.height, window.col_off:window.col_off + window.width]
                print(tile.shape)
                slice_meta = self.ortho_meta.copy()
                slice_meta.update({
                    "height": window.height,
                    "width": window.width,
                    "transform": rasterio.windows.transform(window, self.ortho_meta["transform"])
                })
                # Write the tile to the output directory
                with rasterio.open(os.path.join(output_path, f"output_tile_{i}_{j}.tif"), 'w', **slice_meta) as dst:
                    dst.write(tile)
                print(f"Tile {i}_{j} written to output file")

        
    def combine_tiles(self, tiles):
        """
        Combine the tiles into a single image
        """
        pass
        
    def write_ortho(self, img=None):
        if img is None:
            img = self.field
            # Write the image to the output directory
            with rasterio.open(self.output_file, 'w', **self.ortho_meta) as dst:
                dst.write(img)
            print("Ortho written to output file")




def main():
    ortho_splitter = OrthoSplitter(ortho_file, output_file)
    ortho_splitter.load_ortho()
    ortho_splitter.crop_field()
    ortho_splitter.split_field()
    # ortho_splitter.combine_tiles()
    ortho_splitter.write_ortho()
    print("Ortho split and written to output file")


if __name__ == "__main__":
    main()
