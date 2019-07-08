# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""


import torch
import json
import torchvision.transforms as T
import numpy as np
from PIL import Image
from models import Encoder, DecoderWithAttention
import torch.nn.functional as F

data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'
word_map_file = 'datasets/caption_data/WORDMAP_' + data_name + '.json'
checkpoint = 'logs/tmp/19_checkpoint.pth.tar'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(word_map_file, 'r') as f:
    word_map = json.load(f)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# 载入模型
encoder = Encoder()
decoder = DecoderWithAttention(512, 512, 512, vocab_size)
checkpoint = torch.load(checkpoint)
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
encoder.to(device)
decoder.to(device)
encoder.eval()
decoder.eval()


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
        enc_img_size = encoder_out.shape[1]
        encoder_dim = encoder_out.shape[-1]

        # 拉平 encoder_out
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixel, enc_dim)
        num_pixels = encoder_out.shape[1]

        prev_words = torch.tensor([[word_map['<start>']]], dtype=torch.int64).to(device)

        # 存储完全的句子和得分的list
        complete_seqs = list()
        complete_seqs_scores = list()

        # 开始 decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        while True:
            embeddings = decoder.embedding(prev_words).squeeze(1)  # (1, embed_dim)
            att_weight_encoding, _ = decoder.attention(encoder_out, h)  # [(1, 2048), (1, 196)]
            gate = decoder.sigmoid(decoder.f_beta(h))  # (1, 1, enc_dim)
            att_weight_encoding = att_weight_encoding * gate
            h, c = decoder.decode_step(
                torch.cat([embeddings, att_weight_encoding], dim=1), (h, c)
            )

            # 计算交叉熵得分
            scores = decoder.fc(h) # (1, vocab_size)
            scores = F.softmax(scores, dim=1)  # (1, vocab_size)
            word_score, word_idx = scores.max(1)

            step += 1
            # 判断是否句子生成终止
            if word_idx.item() == word_map['<end>'] or step >= 50:
                break

            complete_seqs.append(rev_word_map[word_idx.item()])
            complete_seqs_scores.append(word_score.item())

            prev_words = word_idx

    caps = ' '.join(complete_seqs) + '.'
    return caps, np.mean(complete_seqs_scores)



if __name__ == '__main__':
    img_path = 'datasets/Flicker8k_Dataset/19212715_20476497a3.jpg'
    ori_img = Image.open(img_path).convert("RGB")
    pred_cap, cap_scores = get_image_caption(ori_img)
    print(pred_cap)
