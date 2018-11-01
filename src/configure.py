N_LABELS = 28

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
N_CHANNELS = 4

N_SPLIT = 8

TEST_TIME_AUGMENT = 1

OPTIMIZER = {0: 'SGD',
             1: 'RMSprop',
             2: 'Adagrad',
             3: 'Adadelta',
             4: 'Adam',
             5: 'Adamax',
             6: 'Adam_AMSGrad',
             7: 'Adam_AMSGrad_LR_0.0001'}

LOSS = {0: 'BCE',
        1: 'FocalLoss',
        2: 'F1Loss'}


FRACTION = [0.36239782, 0.043841336, 0.075268817,
            0.059322034, 0.075268817, 0.075268817,
            0.043841336, 0.075268817, 0.01,
            0.01, 0.01, 0.043841336,
            0.043841336, 0.014198783, 0.043841336,
            0.01, 0.028806584, 0.014198783,
            0.028806584, 0.059322034, 0.01,
            0.126126126, 0.028806584, 0.075268817,
            0.01, 0.222493888, 0.028806584,
            0.01]

AUGMENT_PARAMETERS = {'rotation_range': 180,
                      'width_shift_range': 0.2,
                      'height_shift_range': 0.2,
                      'shear_range': 0.2,
                      'zoom_range': 0.4,
                      'fill_mode': 'nearest',
                      'cval': 0.,
                      'horizontal_flip': True,
                      'vertical_flip': True}

TRAINING_INPUT_DIR = "/home/rwth0233/kaggle_HPAIC/input/train"
TEST_INPUT_DIR = "/home/rwth0233/kaggle_HPAIC/input/test"

SAMPLE_SUBMISSION = "/home/rwth0233/kaggle_HPAIC/input/sample_submission.csv"
TRAINING_DATA_CSV = "/home/rwth0233/kaggle_HPAIC/input/train.csv"

DATA_DIR = "/home/rwth0233/kaggle_HPAIC/data"
TRAINING_DATA = "/home/rwth0233/kaggle_HPAIC/data/train.npz"
TEST_DATA = "/home/rwth0233/kaggle_HPAIC/data/test.npz"

MODEL_WEIGHTS_PATH = "/work/rwth0233/kaggle_HPAIC/model"
MODEL_LOG_PATH = "/home/rs619065/kaggle_HPAIC/logs"
MODEL_ACC_LOSS_PATH = "/home/rs619065/kaggle_HPAIC/acc_loss"


SUBMISSION_PATH = "/home/rs619065/kaggle_HPAIC/submission"

TRAINING_OUTPUT_PATH = "/home/rs619065/kaggle_HPAIC/training"
TEST_OUTPUT_PATH = "/home/rs619065/kaggle_HPAIC/test"

VISUALIZATION_PATH = "/home/rs619065/kaggle_HPAIC/visualization"


LABEL_MAP = {
    0: "Nucleoplasm",
    1: "Nuclear membrane",
    2: "Nucleoli",
    3: "Nucleoli fibrillar center",
    4: "Nuclear speckles",
    5: "Nuclear bodies",
    6: "Endoplasmic reticulum",
    7: "Golgi apparatus",
    8: "Peroxisomes",
    9: "Endosomes",
    10: "Lysosomes",
    11: "Intermediate filaments",
    12: "Actin filaments",
    13: "Focal adhesion sites",
    14: "Microtubules",
    15: "Microtubule ends",
    16: "Cytokinetic bridge",
    17: "Mitotic spindle",
    18: "Microtubule organizing center",
    19: "Centrosome",
    20: "Lipid droplets",
    21: "Plasma membrane",
    22: "Cell junctions",
    23: "Mitochondria",
    24: "Aggresome",
    25: "Cytosol",
    26: "Cytoplasmic bodies",
    27: "Rods & rings"
}
