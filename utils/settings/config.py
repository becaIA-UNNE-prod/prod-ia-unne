import numpy as np
import importlib.util
import sys
from pathlib import Path


def load_module(module_file, module_name):
    # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


AUTHORS = 'Papoutsis I., Sykas D., Zografakis D., Sdraka M.'
DATASET_VERSION = '21.03'
RANDOM_SEED = 16
LICENSES = [
    {
        'url': 'url',
        'id': 1,
        'name': 'name'
    },
]

# Divisors of the Positive Integer 10980:
# 1, 2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 30, 36, 45, 60, 61, 90, 122, 180, 183,
# 244, 305, 366, 549, 610, 732, 915, 1098, 1220, 1830, 2196, 2745, 3660, 5490, 10980
IMG_SIZE = 122

# Total pixels for each resolution for Sentinel2 Data
SENTINEL2_PIXELS = {
    '10': 10980,
    '20': 5490,
    '60': 1830,
}

# Band names and their resolutions
BANDS = {
    'B02': 10, 'B03': 10, 'B04': 10, 'B08': 10,
    'B05': 20, 'B07': 20, 'B06': 20, 'B8A': 20, 'B11': 20, 'B12': 20,
    'B01': 60, 'B09': 60, 'B10': 60
}

# Extract patches based on this band
REFERENCE_BAND = 'B02'

# File to load Mappings from (aka native str to english str)
# You can replace the path to your own mappings, be sure to name mapping dictionary as 'CLASSES_MAPPING'
MAPPINGS_FILE = Path('utils/settings/mappings/mappings_cat.py')
#MAPPINGS_FILE = Path('utils/settings/mappings/mappings_fr.py')
module = load_module(MAPPINGS_FILE, MAPPINGS_FILE.stem)
CLASSES_MAPPING = module.CLASSES_MAPPING
RENAME = module.RENAME
SAMPLE_TILES = module.SAMPLE_TILES
COUNTRY_CODE = module.COUNTRY_CODE

# File to load Encodings from (aka english str to ints)
ENCODINGS_FILE = Path('utils/settings/mappings/encodings_en.py')
module = load_module(ENCODINGS_FILE, ENCODINGS_FILE.stem)
CROP_ENCODING = module.CROP_ENCODING

# Maps arbitrary number of classes to a discrete range of numbers starting from 0

# --- For all classes in the encodings file, uncomment this section ---
# LINEAR_ENCODER = {val: i + 1 for i, val in enumerate(sorted(list(CROP_ENCODING.values())))}
#
# LINEAR_ENCODER[0] = 0
# LINEAR_ENCODER[190] = 0   # 'Other cereals n.e.c.'
# LINEAR_ENCODER[192] = 0   # 'Other'
# LINEAR_ENCODER[219] = 0   # 'Other leafy or stem vegetables n.e.c.'
# LINEAR_ENCODER[227] = 0   # 'Other fruit bearing vegetables n.e.c.'
# LINEAR_ENCODER[236] = 0   # 'Other root bulb or tuberous vegetables n.e.c.'
# LINEAR_ENCODER[250] = 0   # 'Vegetables n.e.c.'
# LINEAR_ENCODER[318] = 0   # 'Other tropical and subtropical fruits n.e.c.'
# LINEAR_ENCODER[325] = 0   # 'Other citrus fruit n.e.c.'
# LINEAR_ENCODER[347] = 0   # 'Other berries'
# LINEAR_ENCODER[357] = 0   # 'Other pome fruits and stone fruits n.e.c.'
# LINEAR_ENCODER[367] = 0   # 'Other nuts n.e.c.'
# LINEAR_ENCODER[380] = 0   # 'Other fruits'
# LINEAR_ENCODER[430] = 0   # 'Other temporary oilseed crops'
# LINEAR_ENCODER[439] = 0   # 'Other temporary oilseed crops n.e.c.'
# LINEAR_ENCODER[444] = 0   # 'Other oleaginous fruits n.e.c.'
# LINEAR_ENCODER[550] = 0   # 'Other roots and tubers n.e.c.'
# LINEAR_ENCODER[615] = 0   # 'Other beverage crops n.e.c.'
# LINEAR_ENCODER[623] = 0   # 'Other temporary spice crops n.e.c.'
# LINEAR_ENCODER[630] = 0   # 'Other permanent spice crops n.e.c.'
# LINEAR_ENCODER[790] = 0   # 'Leguminous crops n.e.c.'
# LINEAR_ENCODER[840] = 0   # 'Other sugar crops n.e.c.'
# LINEAR_ENCODER[900] = 0   # 'Other crops and Classes'
# LINEAR_ENCODER[924] = 0   # 'Other temporary fibre crops'
# LINEAR_ENCODER[931] = 0   # 'Temporary medicinal etc. crops'
# LINEAR_ENCODER[932] = 0   # 'Permanent medicinal etc. crops'
# LINEAR_ENCODER[970] = 0   # 'Other Classes'
# LINEAR_ENCODER[997] = 0   # 'No Data Available'
# LINEAR_ENCODER[980] = 0   # 'Other crops'
# LINEAR_ENCODER[981] = 0   # 'Other crops temporary'
# LINEAR_ENCODER[982] = 0   # 'Other crops permanent'
# LINEAR_ENCODER[998] = 0   # 'Unknown crops'

# --- For the selected classes, uncomment this section ---
SELECTED_CLASSES = [
    120,   # 'Maize'
]

LINEAR_ENCODER = {val: i + 1 for i, val in enumerate(sorted(SELECTED_CLASSES))}
LINEAR_ENCODER[0] = 0

# ---

# Class weights for loss function

# --- Uncomment only the weights of the experiment to be run ---
CLASS_WEIGHTS = {
    0: 0,        # Background/No maíz (ignorado en el loss)
    120: 1.1980338175954461,  # Maíz
}

# ---

# Output DTYPE for patches (bands), range is 0-10k, use smaller int to reduce size required
BAND_OUT_DTYPE = np.uint16

# Output DTYPE for patches (labels/parcels), use bigger ints cause parcels have greater range
LABEL_OUT_DTYPE = np.uint32

# Divider for normalizing tiff data to [0-1] range
NORMALIZATION_DIV = 10000
