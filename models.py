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

        #########################################################################
        # TODO:
        # 定义 encoder 进行图片的特征提取
        # 使用预训练的 ResNet-50 作为 encoder，去掉网络的最后两层
        # 增加 AdaptiveAvgPool2d 将编码的特征变成指定的大小
        #########################################################################
        cnn_ext = ...  # 使用预训练的 resnet-50
        modules = ...  # 去掉网络中的最后两层，考虑使用 list(cnn_ext.children())
        self.cnn_ext = ...  # 使用 nn.Sequential 定义好 encoder

        self.adaptive_pool = ...  # 使用 nn.AdaptiveAvgPool2d 将输出改变到指定的大小
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def forward(self, img):
        out = self.cnn_ext(img)  # [bs, 2048, 8, 8]
        out = self.adaptive_pool(out)  # [bs, 2048, enc_img_size, enc_img_size]
        out = out.permute(0, 2, 3, 1)  # [bs, enc_img_size, enc_img_size, 2048]
        return out

    def freeze_params(self, freeze):
        for p in self.cnn_ext.parameters():
            p.requires_grad = False

        for c in list(self.cnn_ext.children())[5:]:
            for p in c.parameters():
                p.requires_grad = (not freeze)


class AttentionModule(nn.Module):
    """
    Attention Module with Decoder
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: 图片经过 Encoder 之后的特征维度
        :param decoder_dim: 解码器隐含状态 h 的维度
        :param attention_dim: 注意力机制的维度
        """
        super(AttentionModule, self).__init__()
        #########################################################################
        # TODO:
        # 定义注意力机制的网络结构
        # 定义一个线性层将编码的特征维度映射到注意力机制的维度
        # 定义一个线性层将解码器的隐含状态映射到注意力机制的维度
        # 定义一个线性层将将注意力机制的维度映射到 1
        # 定义一个激活函数 ReLU
        # 定义一个激活函数 softmax
        #########################################################################
        self.encoder_att = ...  # Linear, encoder_dim -> attention_dim
        self.decoder_att = ...  # Linear, decoder_dim -> attention_dim
        self.full_att = ...  # Linear, attention_dim -> 1
        self.relu = ...  # relu 激活函数
        self.softmax = ...  # softmax 激活函数, dim=1
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def forward(self, encoder_out, decoder_hidden):
        """
        注意力机制的前向传播过程

        :param encoder_out: 提取的图片特征，大小是 (bs, num_pixels, encoder_dim)
        :param decoder_hidden: 前一步的解码输出，大小是 (bs, decoder_dim)
        :return: 注意力编码的权重矩阵
        """
        #########################################################################
        # TODO:
        # 注意力机制的计算过程
        # 将 encoder_out 和 decoder_hidden 映射到 attention_dim 维度分别得到 att1 和 att2
        # 将 att1 和 att2 加起来，再通过一个线性函数将 attention_dim 映射到 1
        # 通过 softmax 得到注意力矩阵
        #########################################################################
        att1 = ...  # 用 self.encoder_att 作用 encoder_out, (bs, num_pixels, attention_dim)
        att2 = ...  # 用 self.decoder_att 作用 decoder_hidden, (bs, attention_dim)
        att2 = ...  # 使用 unsqueeze 将 att2 的维度从 (bs, attention_dim) -> (bs, 1, attention_dim)
        att = ...  # 将 att1 和 att2 求和，这里利用了 broadcast 的机制, (bs, num_pixels, attention_dim)
        att = ...  # 用 relu 作用 att，提供非线性
        att = ...  # 用 self.full_att 作用 att，将维度映射到 1, (bs, num_pixels, 1)
        att = ...  # 使用 squeeze 将 att 维度从 (bs, num_pixels, 1) -> (bs, num_pixels)
        alpha = ...  # 使用 self.softmax 得到每个 pixel 的权重
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################
        # encoder_out 和注意力矩阵进行加权求和
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(1)  # (bs, encoder_dim)
        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :params attention_dim: 注意力机制的维度
        :params embed_dim: 词向量的维度
        :params decoder_dim: 解码器的维度
        :params vocab_size: 单词总数
        :params encoder_dim: 编码图像的特征维度
        :params dropout: dropout 的比例
        """
        super(DecoderWithAttention, self).__init__()
        # 定义类中的参数
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        # 定义注意力机制
        self.attention = AttentionModule(
            encoder_dim, decoder_dim, attention_dim)

        #########################################################################
        # TODO:
        # 定义好 Decoder 中需要的网络层
        # 这些网络层会在后面的前向传播中使用
        #########################################################################
        self.embedding = ...  # 定义词嵌入 word embedding, (vocab_size, embed_dim) 
        self.dropout = ...  # 定义 dropout
        self.decode_step = ...  # 定义 LSTMCell 作为 Decoder 中的序列模块，输入是 embed + encoder_out
        self.init_h = ...  # 定义线性层将 encoder_out 转换成 hidden state
        self.init_c = ...  # 定义线性层将 encoder_out 转换成 cell state
        self.f_beta = ...  # 定义线性层, decoder_dim -> encoder_dim
        self.sigmoid = ...  # 定义 sigoid 激活函数
        self.fc = ...  # 定义输出的线性层
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

        self.init_weights()

    def init_weights(self):
        """
        初始化 embedding 和 fc 的参数，加快收敛速度
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        给 LSTM 传入初始的 hidden state，其依赖于 Encoder 的输出

        :param encoder_out: 通过 Encoder 之后的特征，维度是 (bs, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        # 对所有的 pxiel 求平均
        mean_encoder_out = encoder_out.mean(dim=1)
        # 线性映射分别得到 hidden state 和 cell state
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lens):
        """
        Decoder 动态图构建的过程
        
        Arguments:
            encoder_out {torch.Tensor} -- 编码之后的特征，维度是 (bs, num_pixels, encoder_dim)
            encoded_captions {torch.Tensor} -- word_map 之后的字幕，维度是 (bs, max_caption_len)
            caption_lens {torch.Tensor} -- 字幕长度，维度是 (bs, 1)
        
        Returns:
            predictions -- 预测的字幕
        """
        batch_size = encoder_out.shape[0]
        encoder_dim = encoder_out.shape[-1]
        vocab_size = self.vocab_size

        # flatten encode_out 特征
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (bs, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # 对输入的字幕长度按照降序排列
        caption_lens, sort_idx = caption_lens.squeeze(
            1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_idx]
        encoded_captions = encoded_captions[sort_idx]

        #########################################################################
        # TODO:
        # 构建 Decoder 的前向传播过程
        # 首先得到输入字幕的词向量，然后在每一个时间步输入 hidden state 和
        # 注意力加权平均的 encoder_out，得到最后的预测输出
        #########################################################################
        embeddings = ...  # 得到 encoded_captions 的词向量, (bs, max_caption_lens, embed_dim)

        # 初始化 LSTM hidden state 和 cell state
        h, c = self.init_hidden_state(encoder_out)

        # 我们不会对 <end> 位置进行解码，所以解码的长度是 caption_lens - 1
        decode_lens = (caption_lens - 1).tolist()

        # 定义存储预测结果和注意力矩阵的空 tensor
        predictions = torch.zeros(batch_size, max(
            decode_lens), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(
            decode_lens), num_pixels).to(device)

        # 在每个时间步，通过注意力矩阵和 decoder 上一步的 hidden state 来生成新的单词
        for t in range(max(decode_lens)):
            # 决定当前时间步的 batch_size，通过 [:batch_size_t] 可以得到当前需要的 tensor
            batch_size_t = sum([l > t for l in decode_lens])  
            # 通过注意力机制得到注意力加权的 encode_out 
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],  h[:batch_size_t])

            gate = ...  # 根据公式计算 soft attention 结果

            attention_weighted_encoding = gate * attention_weighted_encoding

            # 前向传播一个时间步，输入是 embeddings 和 attention_weighted_encoding 合并起来，可以使用 torch.cat
            # hidden state 和 cell state 也需要输入到网络中，注意使用 batch_size_t 取得当前有效的 tensor
            h, c = ...  

            preds = ...  # 对 h 进行 dropout 和全连接层得到预测结果

            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lens, alphas, sort_idx
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

