import os
import pandas as pd
import cv2

csv_file = open("./train_world.csv", 'r')
#line = csv_file.readline()

train_df = pd.read_csv(filepath_or_buffer=csv_file)
total_ids = len(train_df['REFERENCE_ID'].unique())

print(f"Total ids = {total_ids}")


ROOT = "/idiap/resource/database/tinyface/"
TEMP = "/idiap/temp/lluevano/tinyface/"

shape = (112, 112)

# print("Start resizing")
# for img_to_load in train_df.itertuples(index=False):
#     img = cv2.imread(os.path.join(ROOT, img_to_load.PATH))
#     #handle grayscale if any
#     if img.shape[-1] != 3:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     #resize
#     img_resize = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
#     full_path = os.path.join(TEMP, "resized", img_to_load.PATH)
#     iter_path = ["resized"] + (img_to_load.PATH.split('/')[:-1])
#     os.makedirs(os.path.join(TEMP,*iter_path),exist_ok=True)
#     if not cv2.imwrite(full_path, img_resize):
#         raise "Not written " + full_path
train_df['REFERENCE_ID'] = pd.factorize(train_df['REFERENCE_ID'])[0] #label encoding
train_df.to_csv('./tinyface_train_list.lst', header=False, index=False, sep=" ")
csv_file.close()
print("Done!")
