import os

from configure import *
from utils import load_data
from utils import get_target
from generator import ImageDataGenerator
from keras.utils.multi_gpu_utils import multi_gpu_model
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
import numpy as np

from keras.models import load_model

model = load_model("GapNet-PL_KFold_0.h5")

img = load_data(data_path=TRAINING_DATA_1024)
target = get_target()

split_filename = os.path.join(DATA_DIR, "KFold_{}.npz".format(0))
split = np.load(file=split_filename)

train_indexes = split['train_indexes']
test_indexes = split['test_indexes']

valid_generator = ImageDataGenerator(x=img,
                                     y=target,
                                     indexes=train_indexes,
                                     batch_size=32,
                                     shuffle=False,
                                     input_shape=(1024, 1024, 4),
                                     learning_phase=True)

parallel_model = multi_gpu_model(model=model, gpus=2, cpu_merge=False)

parallel_model.compile(optimizer='adam', loss=binary_crossentropy, metrics=[binary_accuracy])

train_loss, train_acc = parallel_model.evaluate_generator(generator=valid_generator,
                                                          workers=16,
                                                          use_multiprocessing=True,
                                                          verbose=1,
                                                          steps=100)

print(train_loss)
print(train_acc)
