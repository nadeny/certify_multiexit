
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from watermarking import Keyprint
import argparse
from einops.layers.torch import Rearrange



def args():
    parser = argparse.ArgumentParser()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    parser.add_argument('--device', default=device,
                        nargs='*')

    hpstr = "Batch size"
    parser.add_argument('--batch', default=400,
                        nargs='*', type=int, help=hpstr)

    hpstr = "epsilon"
    parser.add_argument('--epsilon', default=0.7,
                        nargs='*', type=float, help=hpstr)

    hpstr = "keyprint Number"
    parser.add_argument('--keynum', default=10,
                        nargs='*', type=int, help=hpstr)

    hpstr = "multi-exit threshold"
    parser.add_argument('--threshold', default=0.65,
                        nargs='*', type=float, help=hpstr)

    args = parser.parse_args()
    return args

def get_example(num:int):
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    testset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    data_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=num,
                                              shuffle=True,
                                              num_workers=3)
    source, target = next(iter(data_loader))

    return source, target

def main(arg):
    keyprinter = Keyprint(arg)
    source, target = get_example(num=arg.keynum)

    keyprints = keyprinter.generate_keyprint(source)
    keyprints = Rearrange('b (k c) h w -> b k c h w', c=3)(keyprints)

    logs = keyprinter.verification(keyprints)
    return logs



if __name__ == '__main__':
    main(args())


