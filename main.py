import argparse
import os
from cycle_gan import CycleGan
from torch.backends import cudnn
from data_loader import get_loader


def main(config):
    A_train_loader, B_train_loader, A_test_loader, B_test_loader = get_loader(config)

    if config.mode == 'train':
        solver = CycleGan(config, A_train_loader, B_train_loader)
    elif config.mode == 'sample':
        solver = CycleGan(config, A_test_loader, B_test_loader)


    cudnn.benchmark = True

    # create directories if not exist
    if not os.path.exists('test_results'):
        os.mkdir('test_results')
    if not os.path.exists('test_results/AtoB'):
        os.mkdir('test_results/AtoB')
    if not os.path.exists('test_results/BtoA'):
        os.mkdir('test_results/BtoA')
    if not os.path.exists(config.save_root):
        os.mkdir(config.save_root)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'sample':
        solver.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', required=False, default='load2map/', help='')
    parser.add_argument('--train_subfolder', required=False, default='train', help='')
    parser.add_argument('--test_subfolder', required=False, default='test', help='')
    parser.add_argument('--input_ngc', type=int, default=3, help='input channel for generator')
    parser.add_argument('--output_ngc', type=int, default=3, help='output channel for generator')
    parser.add_argument('--input_ndc', type=int, default=3, help='input channel for discriminator')
    parser.add_argument('--output_ndc', type=int, default=1, help='output channel for discriminator')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nb', type=int, default=6, help='the number of resnet block layer for generator')
    parser.add_argument('--input_size', type=int, default=100, help='input size')
    parser.add_argument('--train_epoch', type=int, default=200, help='train epochs num')
    parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--lambdaA', type=float, default=10, help='lambdaA for cycle loss')
    parser.add_argument('--lambdaB', type=float, default=10, help='lambdaB for cycle loss')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--save_root', required=False, default='results', help='results save path')

    parser.add_argument('--mode', type=str, default='sample')  # train = 학습 모드 / sample = 테스트 모드

    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--sample_step', type=int, default=50)

    config = parser.parse_args()
    print(config)
    main(config)