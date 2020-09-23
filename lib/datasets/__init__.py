"""
TODO: Support more datasets:
  - English Bible Speech Dataset
    https://www.kaggle.com/bryanpark/the-world-english-bible-speech-dataset
  -	CMU Arctic Speech Synthesis Dataset
    http://festvox.org/cmu_arctic/
  - Synthetic Wave Dataset with SoX e.g. [sine, square, triangle, sawtooth, trapezium, exp, brow]
  - VCTK Dataset
  - Voice Conversion Challenge (VCC) 2016 dataset
  - Blizzard dataset
  - JSUT dataset
  - Common Voice dataset
    https://toolbox.google.com/datasetsearch/search?query=text%20speech&docid=sGZ%2FjOYUalNI7AzSAAAAAA%3D%3D
"""
from lib.datasets.lj_speech import LINDA_JOHNSON
from lib.datasets.lj_speech import lj_speech_dataset
from lib.datasets.m_ailabs import ELIZABETH_KLETT
from lib.datasets.m_ailabs import ELLIOT_MILLER
from lib.datasets.m_ailabs import JUDY_BIEBER
from lib.datasets.m_ailabs import m_ailabs_en_uk_elizabeth_klett_speech_dataset
from lib.datasets.m_ailabs import m_ailabs_en_us_elliot_miller_speech_dataset
from lib.datasets.m_ailabs import m_ailabs_en_us_judy_bieber_speech_dataset
from lib.datasets.m_ailabs import m_ailabs_en_us_mary_ann_speech_dataset
from lib.datasets.m_ailabs import MARY_ANN
from lib.datasets.utils import Alignment
from lib.datasets.utils import dataset_generator
from lib.datasets.utils import dataset_loader
from lib.datasets.utils import Example
from lib.datasets.utils import precut_dataset_loader
from lib.datasets.utils import Speaker

# TODO: Consider not using the actors realnames in the codebase in an effort to protect their
# privacy.

HILARY_NORIEGA = Speaker('Hilary Noriega')


def hilary_noriega_speech_dataset(root_directory_name: str = 'hilary_noriega',
                                  gcs_path: str = 'gs://wellsaid_labs_datasets/hilary_noriega',
                                  speaker: Speaker = HILARY_NORIEGA,
                                  **kwargs):
    return dataset_loader(root_directory_name, gcs_path, speaker, **kwargs)


ALICIA_HARRIS = Speaker('Alicia Harris')
MARK_ATHERLAY = Speaker('Mark Atherlay')
SAM_SCHOLL = Speaker('Sam Scholl')

__all__ = [
    'Speaker',
    'Example',
    'Alignment',
    'dataset_generator',
    'dataset_loader',
    'precut_dataset_loader',
    'lj_speech_dataset',
    'm_ailabs_en_us_judy_bieber_speech_dataset',
    'm_ailabs_en_us_mary_ann_speech_dataset',
    'm_ailabs_en_us_elliot_miller_speech_dataset',
    'm_ailabs_en_uk_elizabeth_klett_speech_dataset',
    'ALICIA_HARRIS',
    'ELIZABETH_KLETT',
    'ELLIOT_MILLER',
    'JUDY_BIEBER',
    'LINDA_JOHNSON',
    'MARK_ATHERLAY',
    'MARY_ANN',
    'SAM_SCHOLL',
]
