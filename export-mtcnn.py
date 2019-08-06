from mtcnn_pytorch.src.get_nets import PNet, RNet, ONet
import torch
from PIL import Image
from config import get_config

# model definitions in get_nets.py

if __name__ == '__main__':
    conf = get_config(False)

    device = 'gpu'

    # P-Net
    model = PNet().to(device)  # to device
    model.eval()                    # to eval mode
    example = torch.ones([1, 3, 100, 300]).to(device)

    traced = torch.jit.trace(model, example)
    traced.save(str(conf.save_path/'pnet-gpu.pt'))
    a, b = model(example)

    print('P-Net')
    print('Input size {}'.format(example.size()))  # torch.Size([1, 3, 112, 112])
    print('A size     {}'.format(a.size()))        # torch.Size([1, 4, 51, 51])
    print('B size     {}'.format(b.size()))        # torch.Size([1, 2, 51, 51])

    # R-Net
    model = RNet().to(device)  # to device
    model.eval()                    # to eval mode
    example = torch.ones([1, 3, 24, 24]).to(device)

    traced = torch.jit.trace(model, example)
    traced.save(str(conf.save_path/'rnet-gpu.pt'))
    a, b = model(example)

    print('R-Net')
    print('Input size {}'.format(example.size()))  # torch.Size([1, 3, 24, 24])
    print('A size     {}'.format(a.size()))        # torch.Size([1, 4])
    print('B size     {}'.format(b.size()))        # torch.Size([1, 2])

    # O-Net
    model = ONet().to(device)  # to device
    model.eval()                    # to eval mode
    example = torch.ones([1, 3, 48, 48]).to(device)
    traced = torch.jit.trace(model, example)
    traced.save(str(conf.save_path/'onet-gpu.pt'))
    a, b, c = model(example)

    print('O-Net')
    print('Input size {}'.format(example.size()))  # torch.Size([1, 3, 48, 48])
    print('A size     {}'.format(a.size()))        # torch.Size([1, 10])
    print('B size     {}'.format(b.size()))        # torch.Size([1, 4])
    print('C size     {}'.format(c.size()))        # torch.Size([1, 2])
