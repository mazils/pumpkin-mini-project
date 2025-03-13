from color_segmentation import PumpkinCounter, __DebugLoadTestImage, annotated_path
from orthosplitting import output_path
import numpy as np
import os
import cv2

def main():
    total_pumpkins=0
    blob_count = 0
    for i in range(4):
        for j in range(4):
            filename = os.path.join(output_path, f"output_tile_{i}_{j}.tif")    
            img = __DebugLoadTestImage(filename=filename)
            print(img.shape)
            Pc=PumpkinCounter(verbose=False)
            mins=np.array((1000,1000))
            maxs=np.array((img.shape[0]-1000,img.shape[1]-1000))
            no_of_pumpkins= Pc.ProcessImage(img,mins,maxs)
            print(f"Tile {i}_{j} has {no_of_pumpkins} pumpkins")
            total_pumpkins+=no_of_pumpkins
            annotated_image, blob_count_pumpkin = Pc.processImageContours(img)
            blob_count+=blob_count_pumpkin
            print(f"Tile {i}_{j} has {blob_count_pumpkin} pumpkins")
            cv2.imwrite(os.path.join(annotated_path, f"output_tile_{i}_{j}_annotated.tif"), annotated_image)
    print(f"Total number of pumpkins in the field: {total_pumpkins}")
    print(f"Total number of pumpkins in the field: {blob_count}")
    


if __name__=="__main__":\
    main()



