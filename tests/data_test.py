
import sys
sys.path.append('.')
from dataset import CaptionDataset

dataset = CaptionDataset('./datasets/caption_data', 'flickr8k_5_cap_per_img_5_min_word_freq', split='TRAIN')

from IPython import embed; embed()