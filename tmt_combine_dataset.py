import os
import shutil
from utils import crop_circle, rotate_image, crop_circle_by_warp, fill_balck_circle, is_image

path = "F:\\pud_ploy\\tomato_project"
sub_path = os.listdir(path)

dst_folder_name = "All_with_class"
dst_path =os.path.join(path, dst_folder_name)
try:
    os.mkdir(dst_path)
except:
    pass



for sub in sub_path:
    path_in_sub = os.path.join(path, sub)
    files = os.listdir(path_in_sub)

    for file in files:
        print(file)
        if is_image(file) is not None or file != "result":
            src_path_name = os.path.join(path_in_sub, file)
            dst_path_name = os.path.join(dst_path,sub+"__"+ file)
            try:
                shutil.copy2(src_path_name, dst_path_name)
            except:
                print(f"error file {file} in {src_path_name}")





