from __future__ import print_function, division

import numpy as np
from PIL import Image
from configure import *
import cv2

r_img = np.array(Image.open('/home/rwth0233/HPAIC/input/HPAv18/17960_353_A9_1_red.jpg'))
g_img = np.array(Image.open('/home/rwth0233/HPAIC/input/HPAv18/17960_353_A9_1_green.jpg'))
b_img = np.array(Image.open('/home/rwth0233/HPAIC/input/HPAv18/17960_353_A9_1_blue.jpg'))

print(np.max(r_img[:, :, 0]))
print(np.max(g_img[:, :, 1]))
print(np.max(b_img[:, :, 2]))

raw_data = np.stack([r_img[:, :, 0], g_img[:, :, 1], b_img[:, :, 2]], axis=-1)
img = Image.fromarray(np.uint8(raw_data))
img.save("test.png")


r_img = np.array(Image.open(TEST_INPUT_DIR + '/930451c4-bacb-11e8-b2b8-ac1f6b6435d0_red.tif'))
g_img = np.array(Image.open(TEST_INPUT_DIR + '/930451c4-bacb-11e8-b2b8-ac1f6b6435d0_green.tif'))
b_img = np.array(Image.open(TEST_INPUT_DIR + '/930451c4-bacb-11e8-b2b8-ac1f6b6435d0_blue.tif'))

raw_data = np.stack([r_img, g_img, b_img], axis=-1)
img = Image.fromarray(np.uint8(raw_data))
img.save("train.png")
