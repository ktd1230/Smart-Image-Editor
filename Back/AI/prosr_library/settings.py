import os
BACK_DIR = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR,'prosr_library/data')
MODEL_ROOT = os.path.join(DATA_DIR, 'checkpoints')

OUTPUT_ROOT=os.path.join(BACK_DIR, 'Django/media')