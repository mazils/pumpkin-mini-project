from color_segmentation import PumpkinCounter
from orthosplitting import OrthoSplitter
from file_path import annotated_path, ortho_file, output_file
import numpy as np
import os
import cv2

def main():
    
    total_pumpkins=0
    blob_count = 0
    ortho_splitter = OrthoSplitter(ortho_file, output_file, verbose=False)
    Pc=PumpkinCounter(verbose=True)

    for i in range(4):
        for j in range(4):
            tile, mins , maxs = ortho_splitter.orthosplit_to_image(index=(i, j))
            img =  cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype("float")

            # Apply blur 
            # img = cv2.GaussianBlur(img, (3, 3), 0)
            
            mins=np.array((mins[0],mins[1]))
            maxs=np.array((img.shape[0]-maxs[0],img.shape[1]-maxs[1]))
            
            no_of_pumpkins= Pc.ProcessImage(img,mins,maxs)
            total_pumpkins+=no_of_pumpkins
            
            annotated_image, blob_count_pumpkin = Pc.processImageContours(img)
            blob_count+=blob_count_pumpkin
            
            print(f"Mahal Count: Tile {i}_{j} has {no_of_pumpkins} pumpkins")
            print(f"Blob Count: Tile {i}_{j} has {blob_count_pumpkin} pumpkins")
            
            cv2.imwrite(os.path.join(annotated_path, f"output_tile_{i}_{j}_annotated.tif"), annotated_image)
            
    print(f"Total number of pumpkins in the field: {total_pumpkins}")
    print(f"Total number of pumpkins in the field: {blob_count}")
    


if __name__=="__main__":\
    main()



