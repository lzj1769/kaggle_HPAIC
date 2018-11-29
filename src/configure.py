N_TRAINING = 31072
N_TEST = 11702
N_LABELS = 28

IMAGE_WIDTH_512 = 512
IMAGE_HEIGHT_512 = 512
IMAGE_WIDTH_1024 = 1024
IMAGE_HEIGHT_1024 = 1024
IMAGE_WIDTH_2048 = 2048
IMAGE_HEIGHT_2048 = 2048

N_CHANNELS = 4
K_FOLD = 5

DATA_DIR = "/work/rwth0233/HPAIC/data"

TRAINING_DATA_512 = "/work/rwth0233/HPAIC/data/train_512.npy"
TEST_DATA_512 = "/work/rwth0233/HPAIC/data/test_512.npy"

TRAINING_DATA_1024 = "/work/rwth0233/HPAIC/data/train_1024.npy"
TEST_DATA_1024 = "/work/rwth0233/HPAIC/data/test_1024.npy"

TRAINING_DATA_2048 = "/hpcwork/izkf/projects/SingleCellOpenChromatin/HPAIC/data/train_2048.npy"
TEST_DATA_2048 = "/hpcwork/izkf/projects/SingleCellOpenChromatin/HPAIC/data/test_2048.npy"

TRAINING_INPUT_DIR = "/hpcwork/izkf/projects/SingleCellOpenChromatin/HPAIC/input/train"
TEST_INPUT_DIR = "/hpcwork/izkf/projects/SingleCellOpenChromatin/HPAIC/input/test"

SAMPLE_SUBMISSION = "/home/rs619065/HPAIC/input/sample_submission.csv"
TRAINING_DATA_CSV = "/home/rs619065/HPAIC/input/train.csv"
MODEL_WEIGHTS_PATH = "/home/rs619065/HPAIC/model"

THRESHOLD = [0.5, 0.4, 0.4, 0.4, 0.4, 0.4,
             0.4, 0.4, 0.2, 0.2, 0.2, 0.4,
             0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
             0.4, 0.4, 0.2, 0.4, 0.4, 0.4,
             0.4, 0.4, 0.4, 0.2]

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

MODEL_HISTORY_PATH = "/home/rs619065/HPAIC/history"
MODEL_ACC_LOSS_PATH = "/home/rs619065/HPAIC/acc_loss"

SUBMISSION_PATH = "/home/rs619065/HPAIC/submission"

TRAINING_OUTPUT_PATH = "/home/rs619065/HPAIC/training"
TEST_OUTPUT_PATH = "/home/rs619065/HPAIC/test"

VISUALIZATION_PATH = "/home/rs619065/HPAIC/visualization"

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

LABEL_COUNT = {
    0: 12885,
    1: 1254,
    2: 3621,
    3: 1561,
    4: 1858,
    5: 2513,
    6: 1008,
    7: 2822,
    8: 53,
    9: 45,
    10: 28,
    11: 1093,
    12: 688,
    13: 537,
    14: 1066,
    15: 21,
    16: 530,
    17: 210,
    18: 902,
    19: 1482,
    20: 172,
    21: 3777,
    22: 802,
    23: 2965,
    24: 322,
    25: 8228,
    26: 328,
    27: 11
}
