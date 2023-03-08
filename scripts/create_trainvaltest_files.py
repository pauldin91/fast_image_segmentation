import os
from os import listdir
from os.path import isfile, join

current_path = os.path.dirname(os.path.abspath(__file__))
imgs_path = current_path + "/RGB/"
gt_path = current_path + "/annotations/"
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

print("Images path: " + imgs_path)
print("Ground truth path: " + gt_path)

images_dirs = os.listdir(imgs_path)
gt_dirs = os.listdir(gt_path)

for i_dir in images_dirs:
	f = open(current_path + "/" + i_dir + ".txt", "w+")
	
	images_subdirs = os.listdir(imgs_path + i_dir + "/")
	
	for i_sdir in images_subdirs:
		image_files = os.listdir(imgs_path + i_dir + "/" + i_sdir + "/")
		for i_if in image_files:
			image_file_path = imgs_path + i_dir + "/" + i_sdir + "/" + i_if
			gt_file_path = gt_path + i_dir + "/" + i_sdir + "/" + i_if
			gt_file_path = gt_file_path.replace(".jpg", "Ids.png")
			file_line = image_file_path + "\t" + gt_file_path
			f.write(file_line + "\n")
	
	f.close()
