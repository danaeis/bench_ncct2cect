"""2-D port of upstream gea_gan_model.py (github.com/by-lab/Ea-GANs) — plain
single-direction AtoB translation, matching the paper. The vendored copy this
replaced had been hand-modified to prepend one-hot source/target-phase
condition channels and had no working dataset to feed them; reverted to
upstream's contract (input_A -> fake_B, one target per case) to match every
other model in this benchmark (see BENCHMARK.md) and to the paper Ea-GANs
actually describes. Only two changes from upstream: input_A/input_B are 4-D
(B,C,H,W) instead of upstream's 5-D volumetric tensors (models/networks.py is
the 2-D port these tensors feed), and `.data[0]` -> `.item()` (removed after
torch 0.4; upstream's form raises IndexError on any current torch).
"""
import torch
from collections import OrderedDict
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

        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

        if self.opt.rise_sobelLoss:
            self.sobelLambda = 0
        else:
            self.sobelLambda = self.opt.lambda_sobel

        # load/define networks
        which_netG = opt.which_model_netG
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      which_netG, opt.norm, opt.use_dropout, self.gpu_ids)
        if self.isTrain:
            self.D_channel = opt.input_nc + opt.output_nc
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
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']

        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = self.input_A
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = self.input_B

    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.real_A = self.input_A
            self.fake_B = self.netG.forward(self.real_A)
            self.real_B = self.input_B

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        self.fake_sobel = networks.sobelLayer(self.fake_B)
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        self.real_sobel = networks.sobelLayer(self.real_B).detach()
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
        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_sobelL1 = self.criterionL1(self.fake_sobel, self.real_sobel) * self.sobelLambda
        self.loss_G += self.loss_sobelL1

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
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                        ('G_L1', self.loss_G_L1.item()),
                        ('G_sobelL1', self.loss_sobelL1.item()),
                        ('D_GAN', self.loss_D.item())
                        ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def get_current_img(self):
        return self.get_current_visuals()

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
        self.sobelLambda = self.opt.lambda_sobel / 20 * epochNum
        print('update sobel lambda: %f' % (self.sobelLambda))
