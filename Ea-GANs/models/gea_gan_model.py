import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks as networks


class gea_ganModel(BaseModel):
    def name(self):
        return 'gea_ganModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.batchSize = opt.batchSize
        self.fineSize = opt.fineSize
        
        # define tensors - modified for 3D medical images
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)  # 4D tensor for 2D slices
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)  # 4D tensor for 2D slices

        # Phase condition tensors (3 phases: non-contrast, arterial, venous)
        self.source_phase = None  # Will be set in set_input
        self.target_phase = None  # Will be set in set_input

        if self.opt.rise_sobelLoss:            
            self.sobelLambda = 0
        else:
            self.sobelLambda = self.opt.lambda_sobel


        # load/define networks


        which_netG = opt.which_model_netG
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      which_netG, opt.norm, opt.use_dropout, self.gpu_ids, phase_channels=6)
        if self.isTrain:

            self.D_channel = opt.input_nc + opt.output_nc + 6  # +6 for phase conditions
            use_sigmoid = opt.no_lsgan
            
            self.netD = networks.define_D(self.D_channel, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
                

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
        if not self.isTrain:
            self.netG.eval()

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            if self.opt.labelSmooth:
                self.criterionGAN = networks.GANLoss_smooth(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()


            # initialize optimizers

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG)
            networks.print_network(self.netD)
            print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        
        if self.isTrain:
            # During training, we get a dictionary with the data
            input_A = input['A' if AtoB else 'B']
            input_B = input['B' if AtoB else 'A']
            self.input_A.resize_(input_A.size()).copy_(input_A)
            self.input_B.resize_(input_B.size()).copy_(input_B)
            
            # Create one-hot encoded phase conditions from indices
            batch_size = input_A.size(0)
            self.source_phase = torch.zeros(batch_size, 3)
            self.target_phase = torch.zeros(batch_size, 3)
            
            # Set phases using indices from dataset
            for i in range(batch_size):
                self.source_phase[i, input['source_phase']] = 1
                self.target_phase[i, input['target_phase']] = 1
            
            # Move to GPU if needed
            if len(self.gpu_ids) > 0:
                self.source_phase = self.source_phase.cuda()
                self.target_phase = self.target_phase.cuda()
            
            self.image_paths = input['A_paths' if AtoB else 'B_paths']
        else:
            # During testing, handle single image
            input_A = input['A' if AtoB else 'B']
            input_B = input['B' if AtoB else 'A']
            self.input_A.resize_(input_A.size()).copy_(input_A)
            self.input_B.resize_(input_B.size()).copy_(input_B)
            
            # Create one-hot encoded phase conditions from indices
            batch_size = input_A.size(0)
            self.source_phase = torch.zeros(batch_size, 3)
            self.target_phase = torch.zeros(batch_size, 3)
            
            # Set phases using indices from dataset
            for i in range(batch_size):
                self.source_phase[i, input['source_phase']] = 1
                self.target_phase[i, input['target_phase']] = 1
            
            # Move to GPU if needed
            if len(self.gpu_ids) > 0:
                self.source_phase = self.source_phase.cuda()
                self.target_phase = self.target_phase.cuda()
            
            self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        if self.isTrain:
            self.real_A = Variable(self.input_A)  # [batch_size, 1, H, W]
            self.real_B = Variable(self.input_B)  # [batch_size, 1, H, W]
            
            # Get spatial dimensions from the actual input
            batch_size = self.real_A.size(0)
            h = self.real_A.size(2)
            w = self.real_A.size(3)
            
            # Expand phase conditions to spatial dimensions
            source_phase_expanded = self.source_phase.unsqueeze(2).unsqueeze(3).expand(-1, -1, h, w)
            target_phase_expanded = self.target_phase.unsqueeze(2).unsqueeze(3).expand(-1, -1, h, w)
            
            # Concatenate input image with phase conditions
            input_with_conditions = torch.cat([self.real_A, source_phase_expanded, target_phase_expanded], 1)
            
            # Generate fake image
            self.fake_B = self.netG.forward(input_with_conditions)
            
            # Generate Sobel edges
            self.fake_sobel = networks.sobelLayer(self.fake_B)
            self.real_sobel = networks.sobelLayer(self.real_B).detach()
        else:
            self.real_A = Variable(self.input_A)
            self.real_B = Variable(self.input_B)
            
            # Get spatial dimensions
            batch_size = self.real_A.size(0)
            h = self.real_A.size(2)
            w = self.real_A.size(3)
            
            # Expand phase conditions to spatial dimensions
            source_phase_expanded = self.source_phase.unsqueeze(2).unsqueeze(3).expand(-1, -1, h, w)
            target_phase_expanded = self.target_phase.unsqueeze(2).unsqueeze(3).expand(-1, -1, h, w)
            
            # Concatenate input with phase conditions
            input_with_conditions = torch.cat([self.real_A, source_phase_expanded, target_phase_expanded], 1)
            self.fake_B = self.netG.forward(input_with_conditions)

    def backward_D(self):
        # Fake
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        
        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        
        # Sobel loss
        self.loss_sobelL1 = self.criterionL1(self.fake_sobel, self.real_sobel) * self.sobelLambda
        
        # Total generator loss
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_sobelL1
        self.loss_G.backward()

    def optimize_parameters(self):

        self.forward()

        
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):

        return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                        ('G_L1', self.loss_G_L1.data[0]),
                        ('G_sobelL1', self.loss_sobelL1.data[0]),
                        ('D_GAN', self.loss_D.data[0])
                        ])


    def get_current_visuals(self):
        if self.isTrain:
            # During training, show both phases
            return OrderedDict([
                ('real_A', util.tensor2array(self.real_A[0].data)),  # Show input once
                ('fake_B_arterial', util.tensor2array(self.fake_B[0].data)),
                ('real_B_arterial', util.tensor2array(self.real_B[0].data)),
                ('fake_B_portal', util.tensor2array(self.fake_B[1].data)),
                ('real_B_portal', util.tensor2array(self.real_B[1].data))
            ])
        else:
            # During testing, show single phase
            return OrderedDict([
                ('real_A', util.tensor2array(self.real_A.data)),
                ('fake_B', util.tensor2array(self.fake_B.data)),
                ('real_B', util.tensor2array(self.real_B.data))
            ])

    def get_current_img(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])


        
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])


    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def update_sobel_lambda(self, epochNum):
        self.sobelLambda = self.opt.lambda_sobel/20*epochNum
        print('update sobel lambda: %f' % (self.sobelLambda))
