import os
import shutil
import splitfolders
from sklearn.model_selection import train_test_split

# Paths
dataset_dir = "./Dataset"
data_dir = "./Data"

top_dir = os.path.join(data_dir, "top")
side_dir = os.path.join(data_dir, "side")
grade_a_top_dir = os.path.join(top_dir, "grade_a")
grade_b_top_dir = os.path.join(top_dir, "grade_b")
grade_c_top_dir = os.path.join(top_dir, "grade_c")
grade_reject_top_dir = os.path.join(top_dir, "grade_reject")
grade_a_side_dir = os.path.join(side_dir, "grade_a")
grade_b_side_dir = os.path.join(side_dir, "grade_b")
grade_c_side_dir = os.path.join(side_dir, "grade_c")
grade_reject_side_dir = os.path.join(side_dir, "grade_reject")

#Data folder
os.makedirs(data_dir, exist_ok=True)
#Top and side folder
os.makedirs(top_dir, exist_ok=True)
os.makedirs(side_dir, exist_ok=True)
#Top view
os.makedirs(grade_a_top_dir, exist_ok=True)
os.makedirs(grade_b_top_dir, exist_ok=True)
os.makedirs(grade_c_top_dir, exist_ok=True)
os.makedirs(grade_reject_top_dir, exist_ok=True)
#Side view
os.makedirs(grade_a_side_dir, exist_ok=True)
os.makedirs(grade_b_side_dir, exist_ok=True)
os.makedirs(grade_c_side_dir, exist_ok=True)
os.makedirs(grade_reject_side_dir, exist_ok=True)

# Pemisahan top images dan side images
for foldername in os.listdir(dataset_dir):
  current_dir = os.path.join(dataset_dir, foldername)
  for filename in os.listdir(current_dir):
    if filename.endswith('_s.jpg'):
      shutil.copy(os.path.join(current_dir, filename), os.path.join(side_dir + "/" + foldername, filename))
    elif filename.endswith('_t.jpg'):
      shutil.copy(os.path.join(current_dir, filename), os.path.join(top_dir + "/" + foldername, filename))

#Paths
split_top_dir = "./split/top"
split_side_dir = "./split/side"

os.makedirs(split_top_dir, exist_ok = True)
os.makedirs(split_side_dir, exist_ok = True)

# Split data
for view_dir in [top_dir, side_dir]:
  if(view_dir == top_dir):
    splitfolders.ratio(top_dir, split_top_dir, seed=142, ratio=(.8, .2))
  else:
    splitfolders.ratio(side_dir, split_side_dir, seed=142, ratio=(.8, .2))

shutil.move('./split/top/train', './train/top')
shutil.move('./split/top/val', './test/top')

shutil.move('./split/side/train', './train/side')
shutil.move('./split/side/val', './test/side')