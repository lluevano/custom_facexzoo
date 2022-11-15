import cv2
from PIL import Image
import numpy as np
from torchvision import transforms

convert_tensor = transforms.ToTensor()
tensor_img = convert_tensor(img)

img = Image.open("/idiap/temp/lluevano/tinyface/resized/Training_Set/193/193_10.jpg")

tensor_img = convert_tensor(img)

PDT_out = model.prev_module(tensor_img.reshape(1,3,112,112))

final=PDT_out

final -= final.mean()
final /= final.std()
final *= 64
final += 128

final_img = np.clip(final.detach().numpy(),0,255).astype('uint8')
to_save = final_img.reshape((3,112,112))

to_save = to_save.transpose((1,2,0))
cv2.imwrite("/idiap/home/lluevano/193_10_PDT.jpg",cv2.cvtColor(to_save, cv2.COLOR_RGB2BGR))
