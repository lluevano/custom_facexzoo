import cv2
#import torch
#from PIL import Image
import numpy as np
#from torchvision import transforms


#convert_tensor = transforms.ToTensor()
#img = Image.open("/idiap/home/lluevano/my_databases/tinyface/Training_Set/18/18_1.jpg")
#tensor_img = convert_tensor(img)

img = cv2.imread("/idiap/home/lluevano/my_databases/tinyface/Training_Set/18/18_1.jpg")
tensor_img = torch.from_numpy(img)
tensor_img = tensor_img.type(torch.float)
tensor_img = tensor_img.permute((2,0,1))

PDT_out = model.prev_module(tensor_img.reshape(1,3,32,31))

final=PDT_out

final -= final.mean()
final /= final.std()
final *= 64
final += 128

final_img = np.clip(final.detach().numpy(),0,255).astype('uint8')
to_save = final_img.reshape((3,115,115))

to_save = to_save.transpose((1,2,0))
cv2.imwrite("/idiap/home/lluevano/SR_attempt1_lr0.1.jpg", to_save)
