import sys
sys.path.append('.')
import torch
from models import Encoder


if __name__ == '__main__':
    net = Encoder()
    net = net.cuda()
    x = torch.zeros(32, 3, 224, 224).cuda()
    y = net(x)
    from IPython import embed; embed()