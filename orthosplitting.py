import os
import cv2 as cv
import rasterio
from rasterio.windows import Window
from rasterio.plot import show
from file_path import ortho_path, output_path


ortho_file = ortho_path + "/Gyldensteensvej-9-19-2017-orthophoto.tif"
output_file = output_path + "/field.tif"

class OrthoSplitter:
    def __init__(self, ortho_file, output_file, verbose=False, slices=(4, 4)):
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
        self.slices = slices
        self.index = (0,0)
        self.verbose = verbose

    def load_ortho(self):
        self.dataset = rasterio.open(self.ortho_file)
        self.ortho = self.dataset.read()
        self.ortho_meta = self.dataset.meta
        self.columns = self.dataset.width
        self.rows = self.dataset.height
        self.bands = self.dataset.count
        if self.verbose:
            print("Ortho loaded")    
            print(f"Initial Values = Columns: {self.columns}, Rows: {self.rows}, Bands: {self.bands}")
        
        return self.ortho, self.ortho_meta, self.columns, self.rows, self.bands
    
            
    def crop_field(self, ulc = (2310, 1000), lrc = (5810, 12000)):
        """
        Crop the orthophoto to the field of interest
        """
        # Define the tile coordinates (upper left corner and lower right corner)
        tile_ulc = ulc  # Upper left corner (row, col)
        tile_lrc = lrc  # Lower right corner (row, col)
        
        # Create a window using the tile coordinates
        self.window_location =  Window.from_slices((tile_ulc[0], tile_lrc[0]), (tile_ulc[1], tile_lrc[1]))    
        
        self.field = self.dataset.read(window=self.window_location)
        if self.verbose:
            print("Field cropped")
            print(f"Field shape: {self.field.shape}")
        # Update metadata to reflect the new dimensions
        self.ortho_meta.update({
            "height": self.window_location.height,
            "width": self.window_location.width,
            "transform": rasterio.windows.transform(self.window_location, self.dataset.transform)
        })
        
        return self.field
    
    
    def split_field(self, field = None, slices = (4,4), index = (0, 0)):
        """
        Split the field into smaller tiles
        """
        if field is None:   
            field = self.field
        # Current Shape of the field
        bands, rows, cols = field.shape 
        
        self.slices = slices
        slice_r, slice_c = slices
        
        # Calculate the number of rows and columns for each tile
        row_slices = rows // slice_r
        col_slices = cols // slice_c
        if self.verbose:
            print(row_slices, col_slices)
        
        self.index = index
        row, col = index
        if row == slice_r and col == slice_c:
            row = row -1
            col = col -1
        window = Window.from_slices((row * row_slices, (row + 1) * row_slices), (col * col_slices, (col + 1) * col_slices))
        if self.verbose:
            print(f"Window Created: {window}")
            print(f"Window shape: {window.height, window.width}")
        # Read the tile
        tile = field[:, window.row_off:window.row_off + window.height, window.col_off:window.col_off + window.width]
        
        if self.verbose: 
            print(tile.shape)
        return tile

    def write_tile(self, tile=None, all_tiles=False):
        """
        Write the tile to the output directory
        """
        row, col = self.index
        if tile is None:
            tile = self.field
        if all_tiles:
            row, col = self.slices
            for i in range(row):
                for j in range(col):
                    tile = self.split_field(index=(i, j))
                    with rasterio.open(os.path.join(output_path, f"output_tile_{i}_{j}.tif"), 'w', **self.ortho_meta) as dst:
                        dst.write(tile)
                    print(f"Tile {i}_{j} written to output file")
        else:
            with rasterio.open(os.path.join(output_path, f"output_tile_{row}_{col}.tif"), 'w', **self.ortho_meta) as dst:
                dst.write(tile)
            print(f"Tile written to output file")
            
    
    def combine_tiles(self, tiles=None):
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

    def tile_to_image(self, tile=None):
        """
        Convert the tile to an image
        """

        if tile is None:
            tile = self.field
        
        temp = tile.transpose(1, 2, 0)
        t2 = cv.split(temp)
        img_cv = cv.merge([t2[2], t2[1], t2[0]])
        
        if self.verbose:
            print(f"Tile Shape: {tile.shape}")
            print(f"Image Shape: {img_cv.shape}")
        
        return img_cv

    def orthosplit_to_image(self, index=(0, 0)):
        """
        Convert the orthophoto so slices
        and return a tile as an image
        """
        self.load_ortho()
        if self.verbose: print("Ortho loaded")
        self.crop_field()
        if self.verbose: print("Field cropped")
        tile = self.split_field(index=index)
        if self.verbose: print(f"Tile created: {tile.shape} at index {index}")
        img = self.tile_to_image(tile=tile)
        if self.verbose: print("Tile converted to image")   
        return img

def main():
    ortho_splitter = OrthoSplitter(ortho_file, output_file, verbose=True)
    img = ortho_splitter.orthosplit_to_image()
    cv.imwrite("tile_image.png", img)
    # ortho_splitter.write_tile(all_tiles=True) 
    # ortho_splitter.combine_tiles()


if __name__ == "__main__":
    main()
