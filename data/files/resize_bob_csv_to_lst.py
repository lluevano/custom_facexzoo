import os
import pandas as pd
import cv2

ROOT = "/idiap/resource/database/scface/SCface_database/"
TEMP = "/idiap/temp/lluevano/scface/"

BOB_TRAIN_WORLD = "./scface_train_world.csv"
RESIZED_FOLDER_NAME = "resized"
FINAL_LST_PATH = './scface_train_world.lst'
shape = (112, 112)

csv_file = open(BOB_TRAIN_WORLD, 'r')
train_df = pd.read_csv(filepath_or_buffer=csv_file)
total_ids = len(train_df['REFERENCE_ID'].unique())

print(f"Total ids = {total_ids}")

print("Start resizing")
for img_to_load in train_df.itertuples(index=False):
    img = cv2.imread(os.path.join(ROOT, img_to_load.PATH))
    #handle grayscale if any
    if img.shape[-1] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # resize
    img_resize = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)


    full_path = os.path.join(TEMP, RESIZED_FOLDER_NAME, img_to_load.PATH)
    iter_path = [RESIZED_FOLDER_NAME] + (img_to_load.PATH.split('/')[:-1])
    #os.makedirs(os.path.join(TEMP, *iter_path), exist_ok=True)
    #if not cv2.imwrite(full_path, img_resize):
    #    raise "Not written " + full_path
train_df['REFERENCE_ID'] = pd.factorize(train_df['REFERENCE_ID'])[0] #label encoding

train_df[['PATH','REFERENCE_ID']].to_csv(FINAL_LST_PATH, header=False, index=False, sep=" ")
csv_file.close()
print("Done!")