N_LABELS = 28

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
N_CHANNELS = 4

TEST_SIZE = 0.1

TRAINING_INPUT_DIR = "/home/rwth0233/kaggle_HPAIC/input/train"
TEST_INPUT_DIR = "/home/rwth0233/kaggle_HPAIC/input/test"

SAMPLE_SUBMISSION = "/home/rwth0233/kaggle_HPAIC/input/sample_submission.csv"
TRAINING_LABELS = "/home/rwth0233/kaggle_HPAIC/input/train.csv"

DATA_DIR = "/home/rwth0233/kaggle_HPAIC/data"
TRAINING_DATA_CSV = "/home/rwth0233/kaggle_HPAIC/data/train.csv"
VALIDATION_DATA_CSV = "/home/rwth0233/kaggle_HPAIC/data/validation.csv"
TRAINING_DATA = "/home/rwth0233/kaggle_HPAIC/data/train.npz"
VALIDATION_DATA = "/home/rwth0233/kaggle_HPAIC/data/validation.npz"
TEST_DATA = "/home/rwth0233/kaggle_HPAIC/data/test.npz"

MODEL_PATH = "/work/rwth0233/kaggle_HPAIC/model"

SUBMISSION_PATH = "/home/rs619065/kaggle_HPAIC/submission"
MODEL_ACC_LOSS_PATH = "/home/rs619065/kaggle_HPAIC/acc_loss"
VALIDATION_PATH = "/home/rs619065/kaggle_HPAIC/validation"
TRAINING_PATH = "/home/rs619065/kaggle_HPAIC/training"

VISUALIZATION_PATH = "/home/rs619065/kaggle_HPAIC/visualization"

PREDICTION_PATH = "/home/rwth0233/kaggle_HPAIC/prediction"

TEST_PATH = "/home/rs619065/kaggle_HPAIC/src/test"
GPU_MONITOR_PATH = "/home/rs619065/kaggle_HPAIC/gpu"

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



