N_TRAINING = 105677
N_TRAINING_WITHOUT_EXTERNAL = 31072
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
SINGLE_LABEL_K_FOLD = 2

DATA_DIR = "/work/rwth0233/HPAIC/data"

TRAINING_DATA_512 = "/hpcwork/izkf/projects/SingleCellOpenChromatin/HPAIC/data/train_512.npy"
TRAINING_DATA_1024 = "/hpcwork/izkf/projects/SingleCellOpenChromatin/HPAIC/data/train_1024.npy"

LEAK_FILE = "/home/rs619065/HPAIC/input/TestEtraMatchingUnder_259_R14_G12_B10.csv"
TEST_DATA_512 = "/home/rwth0233/HPAIC/data/test_512.npy"
TEST_DATA_1024 = "/home/rwth0233/HPAIC/data/test_1024.npy"

TRAINING_INPUT_DIR = "/hpcwork/izkf/projects/SingleCellOpenChromatin/HPAIC/input/train"
TEST_INPUT_DIR = "/hpcwork/izkf/projects/SingleCellOpenChromatin/HPAIC/input/test"

SAMPLE_SUBMISSION = "/home/rs619065/HPAIC/input/sample_submission.csv"
TRAINING_DATA_CSV = "/home/rs619065/HPAIC/input/train.csv"
MODEL_WEIGHTS_PATH = "/home/rs619065/HPAIC/model"

MODEL_HISTORY_PATH = "/home/rs619065/HPAIC/history"
MODEL_ACC_LOSS_PATH = "/home/rs619065/HPAIC/acc_loss"

SUBMISSION_PATH = "/home/rs619065/HPAIC/submission"

TRAINING_OUTPUT_PATH = "/home/rs619065/HPAIC/training"
TEST_OUTPUT_PATH = "/home/rs619065/HPAIC/test"

VISUALIZATION_PATH = "/home/rs619065/HPAIC/visualization"

ENSEMBLE_DATA_PATH = "/home/rs619065/HPAIC/ensemble/data"
ENSEMBLE_KERAS_PATH = "/home/rs619065/HPAIC/ensemble/keras"

HPAV18_CSV = "/home/rs619065/HPAIC/input/HPAv18.csv"
HPAV18_DIR = "/home/rwth0233/HPAIC/input/HPAv18"
SUBCELLULAR_LOCATION_CSV = "/home/rs619065/HPAIC/input/subcellular_location.tsv"


NAME_LABEL_DICT = {
    0: 'Nucleoplasm (GO:0005654)',
    1: 'Nuclear membrane (GO:0031965)',
    2: 'Nucleoli (GO:0005730)',
    3: 'Nucleoli fibrillar center (GO:0001650)',
    4: 'Nuclear speckles (GO:0016607)',
    5: 'Nuclear bodies (GO:0016604)',
    6: 'Endoplasmic reticulum (GO:0005783)',
    7: 'Golgi apparatus (GO:0005794)',
    8: 'Peroxisomes (GO:0005777)',
    9: 'Endosomes (GO:0005768)',
    10: 'Lysosomes (GO:0005764)',
    11: 'Intermediate filaments (GO:0045111)',
    12: 'Actin filaments (GO:0015629)',
    13: 'Focal adhesion sites (GO:0005925)',
    14: 'Microtubules (GO:0015630)',
    15: 'Microtubule ends (GO:1990752)',
    16: 'Cytokinetic bridge (GO:0045171)',
    17: 'Mitotic spindle (GO:0072686)',
    18: 'Microtubule organizing center (GO:0005815)',
    19: 'Centrosome (GO:0005813)',
    20: 'Lipid droplets (GO:0005811)',
    21: 'Plasma membrane (GO:0005886)',
    22: 'Cell Junctions (GO:0030054)',
    23: 'Mitochondria (GO:0005739)',
    24: 'Aggresome (GO:0016235)',
    25: 'Cytosol (GO:0005829)',
    26: 'Cytoplasmic bodies (GO:0036464)',
    27: 'Rods & Rings ()',
    28: 'Midbody (GO:0030496)',
    29: 'Cleavage furrow (GO:0032154)',
    30: 'Nucleus (GO:0005634)',
    31: 'Vesicles (GO:0043231)',
    32: 'Midbody ring (GO:0090543)'
}

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
    0: 40958,
    1: 3072,
    2: 10871,
    3: 3329,
    4: 5130,
    5: 5938,
    6: 3725,
    7: 9405,
    8: 217,
    9: 197,
    10: 182,
    11: 2194,
    12: 2233,
    13: 1458,
    14: 2692,
    15: 63,
    16: 1290,
    17: 446,
    18: 1893,
    19: 3672,
    20: 438,
    21: 13809,
    22: 2729,
    23: 10344,
    24: 428,
    25: 37366,
    26: 706,
    27: 127
}

UP_SAMPLING_FACTOR = {
    0: 1,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 2,
    6: 3,
    7: 1,
    8: 1,
    9: 1,
    10: 1,
    11: 5,
    12: 5,
    13: 6,
    14: 5,
    15: 5,
    16: 5,
    17: 5,
    18: 2,
    19: 2,
    20: 2,
    21: 1,
    22: 1,
    23: 1,
    24: 1,
    25: 1,
    26: 1,
    27: 5
}
