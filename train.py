"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import time

import torch
import torchvision.transforms as T
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from dataset import CaptionDataset
from models import DecoderWithAttention, Encoder, device
from utils import *


def main():
    parser = argparse.ArgumentParser(description='caption model')

    parser.add_argument('--save_dir', type=str,
                        default='logs/tmp', help='directory of model save')

    # 数据集参数
    parser.add_argument('--data_folder', type=str, default='./datasets/caption_data',
                        help='caption dataset folder')
    parser.add_argument('--data_name', type=str, default='flickr8k_5_cap_per_img_5_min_word_freq',
                        help='dataset name [coco, flickr8k, flickr30k]')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='training batch size')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='print training state every n times')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of data loader workers ')

    parser.add_argument('--epochs', type=int, default=120,
                        help='total training epochs')
    parser.add_argument('--grad_clip', type=float,
                        default=5., help='number of gradient clip')
    parser.add_argument('--alpha_c', type=float, default=1.,
                        help='ratio of attention matrix')
    parser.add_argument('--encoder_lr', type=float,
                        default=1e-4, help='encoder learning rate')
    parser.add_argument('--decoder_lr', type=float,
                        default=4e-4, help='decoder learning rate')

    # 模型参数
    parser.add_argument('--attention_dim', type=float,
                        default=512, help='dimension of attention')
    parser.add_argument('--embed_dim', type=float, default=512,
                        help='dimension of word embedding')
    parser.add_argument('--decoder_dim', type=float,
                        default=512, help='dimension of decoder')
    parser.add_argument('--dropout', type=float,
                        default=0.5, help='rate of dropout')
    parser.add_argument('-frz', '--freeze_encoder', action='store_true',
                        help='whether freeze encoder parameters')

    args = parser.parse_args()

    mkdir_if_missing(args.save_dir)
    log_path = os.path.join(args.save_dir, 'log.txt')
    with open(log_path, 'w') as f:
        f.write('{}\n'.format(args))

    # TODO:
    # 定义训练集的数据增强操作和验证集的数据增强操作
    # 图片的大小都已经 resize 到 256 x 256
    # 训练集和验证集都只需要将图片转换成 Tensor，然后用 ImageNet 的 mean 和 std 做标准化
    #
    # 提示：可以查看 torchvision.transforms 中的函数来实现数据增强
    #########################################################################
    tfms = T.Compose([])
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    train_dataset = CaptionDataset(args.data_folder, args.data_name, split='TRAIN',
                                   transform=tfms)
    val_dataset = CaptionDataset(args.data_folder, args.data_name, split='VAL',
                                 transform=tfms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    word_map_file = os.path.join(
        args.data_folder, 'WORDMAP_'+args.data_name+'.json')
    with open(word_map_file, 'r') as f:
        word_map = json.load(f)

    # 初始化模型
    encoder = Encoder()
    encoder.freeze_params(args.freeze_encoder)
    decoder = DecoderWithAttention(
        attention_dim=args.attention_dim,
        embed_dim=args.embed_dim,
        decoder_dim=args.decoder_dim,
        vocab_size=len(word_map),
        dropout=args.dropout
    )
    
    #########################################################################
    # TODO:
    # 定义 Encoder 和 Decoder 的优化器
    # 推荐使用 Adam，encoder 和 decoder 的学习率都使用 args 定义好的默认值
    #########################################################################
    encoder_optimizer = ...
    decoder_optimizer = ...
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    # 把模型放到 GPU 上
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    criterion = nn.CrossEntropyLoss()

    train(
        args=args,
        train_loader=train_loader,
        val_loader=val_loader,
        encoder=encoder,
        decoder=decoder,
        criterion=criterion,
        encoder_optimizer=encoder_optimizer,
        decoder_optimizer=decoder_optimizer,
        log_path=log_path
    )


def train(args, train_loader, val_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, log_path):
    best_top5acc = 0
    epochs_since_improvement = 0
    for epoch in range(args.epochs):

        # 如果连续20个epoch模型的性能都没有改善，直接停止训练
        if epochs_since_improvement == 20:
            break
        # 如果连续8个epoch模型的性能都没有改善，进行学习率衰减
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if encoder_optimizer is not None:
                adjust_learning_rate(encoder_optimizer, 0.8)

        encoder.train()
        decoder.train()

        batch_time = AverageMeter()  # 前向和反向传播的时间
        data_time = AverageMeter()  # 数据读取的时间
        losses = AverageMeter()  # 每个单词的损失
        top5accs = AverageMeter()  # top5 准确率

        start = time.time()

        for i, (imgs, caps, caplens) in enumerate(train_loader):
            data_time.update(time.time() - start)

            # 将数据放到GPU上
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # 前向传播
            imgs = encoder(imgs)
            scores, caps_sorted, decode_lens, alphas, sort_idx = decoder(imgs, caps, caplens)

            targets = caps_sorted[:, 1:]

            scores = pack_padded_sequence(
                scores, decode_lens, batch_first=True).data
            targets = pack_padded_sequence(
                targets, decode_lens, batch_first=True).data

            # 计算损失
            loss = criterion(scores, targets)
            loss += args.alpha_c * ((1. - alphas.sum(dim=1))**2).mean()

            # 反向传播
            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            if args.grad_clip is not None:
                nn.utils.clip_grad_value_(decoder.parameters(), args.grad_clip)
                if encoder_optimizer is not None:
                    nn.utils.clip_grad_value_(
                        encoder.parameters(), args.grad_clip)

            # 更新参数
            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

            top5 = accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decode_lens))
            top5accs.update(top5, sum(decode_lens))
            batch_time.update(time.time() - start)

            start = time.time()

            if (i + 1) % args.print_freq == 0:
                print_str = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(train_loader)) + \
                            'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(batch_time=batch_time) + \
                            'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(data_time=data_time) + \
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss=losses) + \
                            'Top-5 Accuracy {top5.val:.2f}% ({top5.avg:.2f}%)\n'.format(
                                top5=top5accs)
                print(print_str)
                with open(log_path, 'a') as f:
                    f.write(print_str)

        print_str = 'Epoch {} End, Time: {:.3f}'.format(
            epoch, batch_time.sum + data_time.sum)
        print(print_str)
        with open(log_path, 'a') as f:
            f.write(print_str)

        val_top5acc = validate(args, val_loader, encoder,
                               decoder, criterion, log_path)
        is_best = val_top5acc > best_top5acc
        best_top5acc = max(val_top5acc, best_top5acc)
        if not is_best:
            epochs_since_improvement += 1
            print_str = '\nEpochs since last improvement: {}'.format(
                epochs_since_improvement)
            print(print_str)
            with open(log_path, 'a') as f:
                f.write(print_str)

        else:
            epochs_since_improvement = 0

        save_checkpoint(args.save_dir, epoch, epochs_since_improvement, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, is_best)


def validate(args, val_loader, encoder, decoder, criterion, log_path):
    losses = AverageMeter()
    top5accs = AverageMeter()

    encoder.eval()
    decoder.eval()

    start = time.time()
    for (imgs, caps, caplens, allcaps) in val_loader:
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        with torch.no_grad():
            imgs = encoder(imgs)
            scores, caps_sorted, decode_lens, alphas, sort_idx = decoder(
                imgs, caps, caplens)
        targets = caps_sorted[:, 1:]

        scores = pack_padded_sequence(
            scores, decode_lens, batch_first=True).data
        targets = pack_padded_sequence(
            targets, decode_lens, batch_first=True).data

        loss = criterion(scores, targets)

        loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        losses.update(loss.item(), sum(decode_lens))
        top5 = accuracy(scores, targets, 5)
        top5accs.update(top5, sum(decode_lens))

    dur_time = time.time() - start
    print_str = 'Validation: \t' + \
        'Time {:.3f}\t'.format(dur_time) + \
        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss=losses) + \
        'Top-5 Accuracy {top5.val:.3f}% ({top5.avg:.3f}%)\n'.format(
            top5=top5accs)
    print(print_str)
    with open(log_path, 'a') as f:
        f.write(print_str)

    return top5accs.avg


if __name__ == '__main__':
    main()
