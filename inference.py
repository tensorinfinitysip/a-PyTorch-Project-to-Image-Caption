# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""


import torch
import json
import torchvision.transforms as T
from PIL import Image
from models import Encoder, DecoderWithAttention

data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'
word_map_file = 'datasets/caption_data/WORDMAP_' + data_name + '.json'
checkpoint = 'logs/tmp/19_checkpoint.pth.tar'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(word_map_file, 'r') as f:
    word_map = json.load(f)

# 载入模型
encoder = Encoder()
decoder = DecoderWithAttention(512, 512, 512, len(word_map))
checkpoint = torch.load(checkpoint)
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
encoder.to(device)
decoder.to(device)
encoder.eval()
decoder.eval()

from IPython import embed; embed()

preprocess = T.Compose([
    T.Resize(size=(256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_image_caption(ori_img):
    img_tensor = preprocess(ori_img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        encoder_out = encoder(img_tensor)  # (1, enc_img_size, enc_img_size, enc_dim)
        complete_seqs, complete_seqs_scores = decoder.predict_seqs(encoder_out)
        # enc_img_size = encoder_out.shape[1]
        # encoder_dim = encoder_out.shape[-1]

        # # 拉平 encoder_out
        # encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixel, enc_dim)
        # num_pixels = encoder_out.shape[1]

        # prev_words = torch.tensor([[word_map['<start>']]], dtype=torch.int64)

        # # 存储完全的句子和得分的list
        # complete_seqs = list()
        # complete_seqs_scores = list()

        # # 开始 decoding
        # step = 1
        # h, c = decoder.init_hidden_state(encoder_out)

        # while True:
        #     embeddings = decoder


if __name__ == '__main__':
    img_path = ''
    ori_img = Image.open(img_path).convert("RGB")
    pred_cap = get_image_caption(ori_img)
    print(pred_cap)
