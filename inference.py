# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""


import json

import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from models import DecoderWithAttention, Encoder

data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'
word_map_file = 'datasets/caption_data/WORDMAP_' + data_name + '.json'
checkpoint = 'logs/tmp/BEST_MODEL.pth.tar'
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
        complete_alphas = list()

        # 开始 decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        while True:
            embeddings = decoder.embedding(prev_words).squeeze(1)  # (1, embed_dim)
            att_weight_encoding, alpha = decoder.attention(encoder_out, h)  # [(1, 2048), (1, 196)]
            alpha = alpha.view(enc_img_size, enc_img_size)
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
            complete_alphas.append(alpha.cpu().numpy())

            prev_words = word_idx

    return complete_seqs, np.mean(complete_seqs_scores), complete_alphas 



def visualization_att(image, caps, alphas):
    """
    可视化字幕中每一个单词的注意力热力图
    """
    image = image.resize([14*24, 14*24], Image.LANCZOS)
    plt.figure(figsize=(18, 16))
    caps = ['<start>'] + caps
    for t in range(len(caps)):
        plt.subplot(np.ceil(len(caps) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (caps[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        plt.axis('off')

        if t == 0:
            continue
        current_alpha = alphas[t-1]
        alpha = cv2.resize(current_alpha, dsize=None, fx=24, fy=24)
        alpha = cv2.GaussianBlur(alpha,(7, 7),0)
        plt.imshow(alpha, alpha=0.6)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')

    plt.show()


if __name__ == '__main__':
    img_path = 'datasets/Flicker8k_Dataset/19212715_20476497a3.jpg'
    ori_img = Image.open(img_path).convert("RGB")
    pred_caps, cap_scores, alphas = get_image_caption(ori_img)
    visualization_att(ori_img, pred_caps, alphas)
    caps = ' '.join(pred_caps) + '.'
    print(caps)
