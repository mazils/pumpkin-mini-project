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
        self.slice_meta = None  
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
    
    
    def split_field(self, field=None, slices=(4, 4), index=(0, 0), overlap=0):
        """
        Split the field into smaller tiles with optional overlap.
        Overlap should be specified as a percentage between 0 and 100.
        """
        if not (0 <= overlap <= 100):
            raise ValueError("Overlap must be between 0 and 100")

        overlap /= 100  # Convert percentage to a fraction

        if field is None:
            field = self.field

        # Current Shape of the field
        bands, rows, cols = field.shape

        slice_r, slice_c = slices
        self.slices = slices
        
        # Calculate the number of rows and columns for each tile
        row_slices = rows // slice_r
        col_slices = cols // slice_c

        overlap_r = int(row_slices * overlap)
        overlap_c = int(col_slices * overlap)

        self.index = index
        row, col = index

        if row >= slice_r or col >= slice_c:
            print("Invalid index")
            return None
        

        
        top_row = row * row_slices
        bottom_row = (row + 1) * row_slices
        left_col = col * col_slices
        right_col = (col + 1) * col_slices
                
        # Create a window for the tile
        overlap_r_top = 0 if row == 0 else -overlap_r
        overlap_r_bottom = 0 if row == slice_r - 1 else overlap_r
        overlap_c_left = 0 if col == 0 else -overlap_c
        overlap_c_right = 0 if col == slice_c - 1 else overlap_c

        window = Window.from_slices((top_row + overlap_r_top,
                                     bottom_row + overlap_r_bottom), 
                                    (left_col + overlap_c_left,
                                     right_col + overlap_c_right))
        
        # Read the tile
        tile = field[:, window.row_off:window.row_off + window.height ,window.col_off  :window.col_off + window.width]   

        slice_meta = self.ortho_meta.copy()
        self.slice_meta = slice_meta.copy()
        slice_meta.update({
            "height": window.height,
            "width": window.width,
            "transform": rasterio.windows.transform(window, self.ortho_meta["transform"])
        })
        self.slice_meta = slice_meta.copy()

        if self.verbose:
            print(f"Window Created: {window}")
            print(f"Row slices: {row_slices}, Column slices: {col_slices}")
            print(f"Overlap rows: {overlap_r}, Overlap columns: {overlap_c}")
            # print(f"Window shape: {window.height, window.width}")
            # print(f"Window offset: {window.row_off, window.col_off}")

        return tile

    def write_tile(self, tile=None, all_tiles=False, output_path=output_path, overlap=0):
        """
        Write the tile to the output directory
        """
        row, col = self.index
        if tile is None:
            tile = self.field
            print(f"Tile not specified. Using the cropped field")
        if all_tiles:
            row, col = self.slices
            for i in range(row):
                for j in range(col):
                    tile = self.split_field(index=(i, j),overlap=overlap)
                    with rasterio.open(os.path.join(output_path, f"output_tile_{i}_{j}.tif"), 'w', **self.slice_meta) as dst:
                        dst.write(tile)
                    print(f"Tile {i}_{j} written to output file")
        else:
            with rasterio.open(os.path.join(output_path, f"output_tile_{row}_{col}.tif"), 'w', **self.slice_meta) as dst:
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
    
        # img_tiff = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB).astype("float")

        if self.verbose:
            print(f"Tile Shape: {tile.shape}")
            print(f"Image Shape: {img_cv.shape}")
        
        return img_cv

    def orthosplit_to_image(self, index=(0, 0), overlap=0):
        """
        Convert the orthophoto so slices
        and return a tile as an image
        """
        self.load_ortho()
        if self.verbose: print("Ortho loaded")
        self.crop_field()
        if self.verbose: print("Field cropped")
        tile = self.split_field(index=index, overlap=overlap)   
        if self.verbose: print(f"Tile created: {tile.shape} at index {index} with overlap {overlap}%")   
        img = self.tile_to_image(tile=tile)
        if self.verbose: print("Tile converted to image")   
        return img

def main():
    ortho_splitter = OrthoSplitter(ortho_file, output_file, verbose=True)
    img = ortho_splitter.orthosplit_to_image(overlap=5, index=(1, 1))
    cv.imwrite("tile_image.png", img)
    ortho_splitter.write_tile(all_tiles=False, overlap=2.5) 
    # ortho_splitter.combine_tiles()


if __name__ == "__main__":
    main()
