# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""


import torch
import torchvision.transforms as T
from models import Encoder, DecoderWithAttention

data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'
word_map_file = 'datasets/WORDMAP_' + data_name + '.json'
checkpoint = 'logs/tmp/BEST_checkpoint.pth.tar'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(word_map_file, 'r') as f:
    word_map = json.load(f)

# 载入模型
encoder = Encoder()
decoder = DecoderWithAttention(512, 512, 512, len(word_map), 0.5)
checkpoint = torch.load(checkpoint)
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])

encoder.eval()