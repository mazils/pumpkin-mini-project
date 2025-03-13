import os

input_rel = "../figures"

script_dir = os.path.dirname(__file__)

input_rel = "../figures"
ortho_rel = "../visualized-qgis-images"
annotated_rel = "../annotated-images"  
output_rel = "../processed-images"

input_path = os.path.join(script_dir, input_rel)
output_path = os.path.join(script_dir, output_rel)  
ortho_path = os.path.join(script_dir, ortho_rel)
annotated_path = os.path.join(script_dir, annotated_rel)

# Create directories if they do not exist
os.makedirs(input_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)
os.makedirs(ortho_path, exist_ok=True)
os.makedirs(annotated_path, exist_ok=True)

ortho_file = ortho_path + "/Gyldensteensvej-9-19-2017-orthophoto.tif"
output_file = output_path + "/field.tif"

    # image_name = "EB-02-660_0595_0435"
    # image = "Data/Images/" + image_name + ".JPG"
    # image = "./Gyldensteensvej-9-19-2017-orthophoto.tif"