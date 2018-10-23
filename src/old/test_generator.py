from __future__ import print_function, division

from generator import ImageDataGenerator
from configure import *
from PIL import Image

from utils import *

OUTPUT_DIR = "/home/rs619065/kaggle_HPAIC/src/test"

valid_img, valid_label = load_data(dataset="validation")
valid_generator = ImageDataGenerator(images=valid_img, augment=True,
                                     horizontal_flip=True, vertical_flip=True,
                                     rescale=1)

print("testing horizontal and vertical flip...", file=sys.stderr)

df = pd.read_csv(VALIDATION_DATA_CSV)

img = valid_generator[0]

for i in range(img.shape[0]):
    prefix = df.iloc[i][0]
    r_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_red.png".format(prefix)))
    g_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_green.png".format(prefix)))
    b_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_blue.png".format(prefix)))
    y_img = Image.open(os.path.join(TRAINING_INPUT_DIR, "{}_yellow.png".format(prefix)))

    raw_data = np.stack([r_img, g_img, b_img, y_img], axis=-1)
    im_raw = Image.fromarray(np.uint8(raw_data))
    im_aug = Image.fromarray(np.uint8(img[i]))

    im_raw.save(fp=os.path.join(OUTPUT_DIR, "{}_raw.png".format(prefix)))
    im_aug.save(fp=os.path.join(OUTPUT_DIR, "{}_raw_flip.png".format(prefix)))