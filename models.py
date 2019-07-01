"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
from torchvision.models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_img_size = encoded_image_size

        cnn_ext = resnet50(pretrained=True)

        modules = list(cnn_ext.children())[:-2]
        self.cnn_ext = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size))

        self.freeze_params(freeze=True)

    def forward(self, img):
        out = self.cnn_ext(img)  # [bs, 2048, h, w]
        out = self.adaptive_pool(out)  # [bs, 2048, enc_img_size, enc_img_size]
        out = out.permute(0, 2, 3, 1)  # [bs, enc_img_size, enc_img_size, 2048]
        return out

    def freeze_params(self, freeze=True):
        for p in self.cnn_ext.parameters():
            p.requires_grad = False

        for c in list(self.cnn_ext.children())[5:]:
            for p in c.parameters():
                p.requires_grad = freeze


class AttentionModule(nn.Module):
    """
    Attention Module with Decoder
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: 编码器的维度
        :param decoder_dim: 解码器的维度
        :param attention_dim: 注意力机制的维度
        """
        super(AttentionModule, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU(True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        注意力机制的前向传播

        :param encoder_out: 提取的图片特征，大小是 [bs, num_pixels, encoder_dim]
        :param decoder_out: 前一步的解码输出，大小是 [bs, decoder_dim]
        :return: 注意力编码的权重矩阵
        """
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # [bs, num_pixels]
        alpha = self.softmax(att)  # [bs, num_pixels]
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(1)
        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :params attention_dim: 注意力的维度
        :params embed_dim: 词向量的维度
        :params decoder_dim: 解码器的维度
        :params vocab_size: 单词总数
        :params encoder_dim: 编码图像的特征维度
        :params dropout: dropout 的比例
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        self.attention = AttentionModule(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.decode_step = nn.LSTMCell(embed_dim+encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        """
        初始化一些参数，加快收敛
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        给 LSTM 传入初始的 hidden state，其依赖于 encoder 的输出
        
        :param encoder_out: 通过卷积网络得到的编码之后的图像，大小是 [bs, num_pixels, encoder_dim]
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lens):
        """
        Decoder 计算图构建

        :param encoder_out: 通过卷积网络得到的编码之后的图像，大小是 [bs, num_pixels, encoder_dim]
        :param encoded_captions: 被编码之后的字幕，大小是 [bs, max_caption_len]
        :param caption_lens: 字幕长度，大小是 [bs, 1]
        :return: 
        """
        batch_size = encoder_out.shape[0]
        encoder_dim = encoder_out.shape[-1]
        vocab_size = self.vocab_size

        # 拉平图片特征
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # [bs, num_pixels, encoder_dim]
        num_pixels = encoder_out.size(1)

        # 对输入的字幕长度按照降序排列
        caption_lens, sort_idx = caption_lens.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_idx]
        encoded_captions = encoded_captions[sort_idx]

        # 得到词向量
        embeddings = self.embedding(encoded_captions) # [bs, max_caption_lens, embed_dim]

        # 初始化 LSTM hidden state
        h, c = self.init_hidden_state(encoder_out)

        # 我们不会对 <end> 位置进行解码，所以解码的长度是 caption_lens - 1
        decode_lens = (caption_lens - 1).tolist()

        #
        predictions = torch.zeros(batch_size, max(decode_lens), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lens), num_pixels).to(device)

        # 在每个时间步，通过注意力矩阵和 decoder 上一步的 hidden state 来生成新的单词
        for t in range(max(decode_lens)):
            batch_size_t = sum([l > t for l in decode_lens])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lens, alphas, sort_idx