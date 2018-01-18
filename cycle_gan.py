import torch
import torch.nn as nn
import torchvision
import os
import pickle
import scipy.io
import numpy as np

from torch.autograd import Variable
from torch import optim
from network import G
from network import D


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

class CycleGan(object):
    def __init__(self, config, A_data_loader, B_data_loader):
        self.A_data_loader = A_data_loader
        self.B_data_loader = B_data_loader

        self.G_A = None
        self.G_B = None
        self.D_A = None
        self.D_B = None

        self.input_ngc = config.input_ngc
        self.output_ngc = config.output_ngc
        self.input_ndc = config.input_ndc
        self.output_ndc = config.output_ndc
        self.batch_size = config.batch_size
        self.ngf = config.ngf
        self.ndf = config.ndf
        self.nb = config.nb
        self.input_size = config.input_size

        self.train_epoch = config.train_epoch
        self.lrD = config.lrD
        self.lrG = config.lrG
        self.lambdaA = config.lambdaA
        self.lambdaB = config.lambdaB
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.save_root = config.save_root

        self.log_step = config.log_step
        self.sample_step = config.sample_step

        self.build_model()

    def build_model(self):
        self.G_A = G(self.input_ngc, self.output_ngc, self.ngf, self.nb)
        self.G_B = G(self.input_ngc, self.output_ngc, self.ngf, self.nb)
        self.D_A = D(self.input_ndc, self.output_ndc, self.ndf)
        self.D_B = D(self.input_ndc, self.output_ndc, self.ndf)

        self.G_A.weight_init(mean=0.0, std=0.02)
        self.G_B.weight_init(mean=0.0, std=0.02)
        self.D_A.weight_init(mean=0.0, std=0.02)
        self.D_B.weight_init(mean=0.0, std=0.02)

        # loss
        self.BCE_loss = nn.BCELoss()
        self.MSE_loss = nn.MSELoss()
        self.L1_loss  = nn.L1Loss()

        # optimizer
        g_params = list(self.G_A.parameters()) + list(self.G_B.parameters())
        self.g_optimizer  = optim.Adam(g_params, self.lrG, [self.beta1, self.beta2])
        self.dA_optimizer = optim.Adam(self.D_A.parameters(), self.lrD, [self.beta1, self.beta2])
        self.dB_optimizer = optim.Adam(self.D_B.parameters(), self.lrD, [self.beta1, self.beta2])

        if torch.cuda.is_available():
            self.G_A.cuda()      # realB 를 넣었을 때, 정교한 A 타입 이미지 생성
            self.G_B.cuda()      # realA 를 넣었을 때, 정교한 B 타입 이미지 생성
            self.D_A.cuda()      # 들어온 이미지가 실제 A 타입 이미지와 얼마나 비슷한지 결정
            self.D_B.cuda()      # 들어온 이미지가 실제 B 타입 이미지와 얼마나 비슷한지 결정
            self.BCE_loss.cuda()
            self.MSE_loss.cuda()
            self.L1_loss.cuda()

        print('---------- Networks initialized -------------')
        print_network(self.G_A)
        print_network(self.G_B)
        print_network(self.D_A)
        print_network(self.D_B)
        print('---------------------------------------------')

    def to_variable(self, x):
        """Convert tensor to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def denorm(self, x):
        """Convert range (-1, 1) to (0, 1)"""
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def train(self):
        total_steps = min(len(self.A_data_loader), len(self.B_data_loader))

        for epoch in range(self.train_epoch):
            current_step = 0

            for images_A, images_B in zip(self.A_data_loader, self.B_data_loader):
                realA = self.to_variable(images_A)
                realB = self.to_variable(images_B)

                """ Train G """
                fake_image_A    = self.G_A(realB)
                D_A_result      = self.D_A(fake_image_A)
                G_A_loss        = self.MSE_loss(D_A_result, self.to_variable(torch.ones(D_A_result.size())))

                reconstructionB = self.G_B(fake_image_A)
                A_cycle_loss    = self.L1_loss(reconstructionB, realB) * self.lambdaB

                fake_image_B    = self.G_B(realA)
                D_B_result      = self.D_B(fake_image_B)
                G_B_loss        = self.MSE_loss(D_B_result, self.to_variable(torch.ones(D_B_result.size())))

                reconstructionA = self.G_A(fake_image_B)
                B_cycle_loss    = self.L1_loss(reconstructionA, realA) * self.lambdaA

                G_loss = G_A_loss + A_cycle_loss + G_B_loss + B_cycle_loss

                self.g_optimizer.zero_grad()
                G_loss.backward()
                self.g_optimizer.step()

                """ Train D_A """
                D_A_real        = self.D_A(realA)
                D_A_real_loss   = self.MSE_loss(D_A_real, self.to_variable(torch.ones(D_A_real.size())))

                fake_image_A    = self.G_A(realB)
                D_A_result      = self.D_A(fake_image_A)
                D_A_fake_loss   = self.MSE_loss(D_A_result, self.to_variable(torch.zeros(D_A_result.size())))

                D_A_loss        = (D_A_real_loss + D_A_fake_loss) * 0.5

                self.dA_optimizer.zero_grad()
                D_A_loss.backward()
                self.dA_optimizer.step()

                """ Train D_B """
                D_B_real = self.D_B(realB)
                D_B_real_loss = self.MSE_loss(D_B_real, self.to_variable(torch.ones(D_B_real.size())))

                fake_image_B = self.G_B(realA)
                D_B_result = self.D_B(fake_image_B)
                D_B_fake_loss = self.MSE_loss(D_B_result, self.to_variable(torch.zeros(D_B_result.size())))

                D_B_loss = (D_B_real_loss + D_B_fake_loss) * 0.5

                self.dB_optimizer.zero_grad()
                D_B_loss.backward()
                self.dB_optimizer.step()

                """ print log """
                if(current_step + 1) % self.log_step == 0:
                    print('Epoch [%d/%d], Step[%d/%d] - loss_D_A: %.3f, loss_D_B: %.3f, '
                          'loss_G_A: %.3f, loss_G_B: %.3f, loss_A_cycle: %.3f, loss_B_cycle: %.3f'
                          % ((epoch + 1), self.train_epoch, current_step + 1, total_steps,
                            D_A_loss.data[0], D_B_loss.data[0],
                            G_A_loss.data[0], G_B_loss.data[0], A_cycle_loss.data[0], B_cycle_loss.data[0]))

                """ save sample image """
                if (current_step + 1) % self.sample_step == 0:
                    genA = self.G_A(realB)
                    genB = self.G_B(realA)

                    path = 'test_results/' + '/AtoB/'
                    torchvision.utils.save_image(self.denorm(genB.data),
                                                 os.path.join(path, 'fake_samples-%d-%d.png' % (epoch + 1, current_step + 1)))

                    path = 'test_results/' + '/BtoA/'
                    torchvision.utils.save_image(self.denorm(genA.data),
                                                 os.path.join(path, 'fake_samples-%d-%d.png' % (epoch + 1, current_step + 1)))

                current_step += 1

            # save the model parameters for each epoch
            gA_path = os.path.join(self.save_root, 'generatorA-%d.pkl' % (epoch + 1))
            gB_path = os.path.join(self.save_root, 'generatorB-%d.pkl' % (epoch + 1))
            dA_path = os.path.join(self.save_root, 'discriminatorA-%d.pkl' % (epoch + 1))
            dB_path = os.path.join(self.save_root, 'discriminatorB-%d.pkl' % (epoch + 1))

            torch.save(self.G_A.state_dict(), gA_path)
            torch.save(self.G_B.state_dict(), gB_path)
            torch.save(self.D_A.state_dict(), dA_path)
            torch.save(self.D_B.state_dict(), dB_path)

    def sample(self):
        # Load trained parameters
        gA_path = os.path.join(self.save_root, 'generatorA-%d.pkl' % (self.train_epoch))
        gB_path = os.path.join(self.save_root, 'generatorB-%d.pkl' % (self.train_epoch))
        dA_path = os.path.join(self.save_root, 'discriminatorA-%d.pkl' % (self.train_epoch))
        dB_path = os.path.join(self.save_root, 'discriminatorB-%d.pkl' % (self.train_epoch))

        self.G_A.load_state_dict(torch.load(gA_path))
        self.G_B.load_state_dict(torch.load(gB_path))
        self.D_A.load_state_dict(torch.load(dA_path))
        self.D_B.load_state_dict(torch.load(dB_path))

        self.G_A.eval()
        self.G_B.eval()
        self.D_A.eval()
        self.D_B.eval()

        # Sample the images
        images_A, images_B = list(zip(self.A_data_loader, self.B_data_loader))[0]
        realA = self.to_variable(images_A)
        realB = self.to_variable(images_B)

        genA = self.G_A(realB)
        genB = self.G_B(realA)

        path = 'test_results/' + '/AtoB/'
        sample_path = os.path.join(path, 'fake_samples-final_AtoB.png')
        torchvision.utils.save_image(self.denorm(genB.data), sample_path, nrow=12)
        print("Saved sampled images (A to B) to '%s'" % path)

        path = 'test_results/' + '/BtoA/'
        sample_path = os.path.join(path, 'fake_samples-final_BtoA.png')
        torchvision.utils.save_image(self.denorm(genA.data), sample_path, nrow=12)
        print("Saved sampled images (B to A) to '%s'" % path)
