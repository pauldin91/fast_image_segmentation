# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


from config import config
from base_model import resnet18
from seg_opr.seg_oprs import ConvBnRelu, AttentionRefinement, FeatureFusion
from pix2pix_networks import define_G, define_D, GANLoss


def get():
    return BiSeNet(config.num_classes, None, None)

class BiSeNet_crowd(nn.Module):
    def __init__(self, out_planes, is_training,
                 criterion, ohem_criterion, pretrained_model=None,
                 norm_layer=nn.BatchNorm2d):
        super(BiSeNet_crowd, self).__init__()
        self.context_path = resnet18(pretrained_model, norm_layer=norm_layer,
                                     bn_eps=config.bn_eps,
                                     bn_momentum=config.bn_momentum,
                                     deep_stem=False, stem_width=64)

        self.business_layer = []
        self.is_training = is_training

        self.spatial_path = SpatialPath(3, 128, norm_layer)

        self.i2i_path = define_G(input_nc=512, output_nc=3, ngf=64, netG='resnet_4blocks', norm='batch', use_dropout=True, init_type='normal', init_gain=0.02, gpu_ids=[0])

        i2i_arms = [AttentionRefinement(256, 128, norm_layer),
                    AttentionRefinement(128, 128, norm_layer)]
        i2i_ffms = [FeatureFusion(128 * 2, 128, 1, norm_layer),
                    FeatureFusion(128 * 2, 128, 1, norm_layer)]

        conv_channel = 128
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(512, conv_channel, 1, 1, 0,
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer)
        )

        arms = [AttentionRefinement(512, conv_channel, norm_layer),
                AttentionRefinement(256, conv_channel, norm_layer)]

        refines = [ConvBnRelu(conv_channel, conv_channel, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, has_bias=False),
                   ConvBnRelu(conv_channel, conv_channel, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, has_bias=False)]

        if is_training:
            heads = [BiSeNetHead(conv_channel, out_planes, 2,
                                 True, norm_layer),
                     BiSeNetHead(conv_channel, out_planes, 1,
                                 True, norm_layer),
                     BiSeNetHead(conv_channel * 2, out_planes, 1,
                                 False, norm_layer)]
        else:
            heads = [None, None,
                     BiSeNetHead(conv_channel * 2, out_planes, 1,
                                 False, norm_layer)]

        self.ffm = FeatureFusion(conv_channel * 2, conv_channel * 2, 1, norm_layer)
        self.arms = nn.ModuleList(arms)
        self.refines = nn.ModuleList(refines)
        self.heads = nn.ModuleList(heads)

        self.i2i_arms = nn.ModuleList(i2i_arms)
        self.i2i_ffms = nn.ModuleList(i2i_ffms)

        self.business_layer.append(self.spatial_path)
        self.business_layer.append(self.global_context)
        self.business_layer.append(self.arms)
        self.business_layer.append(self.i2i_arms)
        self.business_layer.append(self.i2i_ffms)
        self.business_layer.append(self.refines)
        self.business_layer.append(self.heads)
        self.business_layer.append(self.ffm)

        if is_training:
            self.criterion = criterion
            self.ohem_criterion = ohem_criterion

            self.netD = define_D(input_nc=6, ndf=32, netD='basic', n_layers_D=2, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[0])
            self.device = torch.device(config.gpu_number if torch.cuda.is_available() else "cpu")
            self.criterionGAN = GANLoss(gan_mode='vanilla', target_real_label=1.0, target_fake_label=0.0).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=0.000002, betas=(0.5, 0.999))

    def forward(self, data):
        spatial_out = self.spatial_path(data)

        self.data = data

        self.data_resized = F.interpolate(self.data,
                                       size=(self.data.shape[2:][0] // 8, self.data.shape[2:][1] // 8),
                                       mode='bilinear', align_corners=True)

        context_blocks = self.context_path(self.data)
        context_blocks.reverse()

        i2i_blocks = self.i2i_path(context_blocks[0])

        global_context = self.global_context(context_blocks[0])
        global_context = F.interpolate(global_context,
                                       size=context_blocks[0].size()[2:],
                                       mode='bilinear', align_corners=True)

        last_fm = global_context
        self.pred_out = []

        for i, (fm, arm, i2i_arm, refine, i2i_ffm) in enumerate(zip(context_blocks[:2], self.arms, self.i2i_arms,
                                                  self.refines, self.i2i_ffms)):
            fm = arm(fm)
            i2i = i2i_arm(i2i_blocks[i])
            fm += last_fm
            last_fm = F.interpolate(fm, size=(context_blocks[i + 1].size()[2:]),
                                    mode='bilinear', align_corners=True)
            last_fm = i2i_ffm(last_fm, i2i)
            last_fm = refine(last_fm)
            self.pred_out.append(last_fm)
        context_out = last_fm

        concate_fm = self.ffm(spatial_out, context_out)
        self.pred_out.append(concate_fm)

        self.pred_out.append(i2i_blocks[-2])
        self.pred_out.append(i2i_blocks[-1])

        if not self.is_training:
            return F.log_softmax(self.heads[-1](self.pred_out[2]), dim=1)

    def backward_D(self):

        fake_pair = torch.cat((self.data_resized, self.pred_out[-1]), 1)
        pred_fake = self.netD(fake_pair.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_pair = torch.cat((self.data_resized, self.i2i_label), 1)
        pred_real = self.netD(real_pair)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

        return self.loss_D

    def backward_G(self):

        fake_pair = torch.cat((self.data, self.pred_out[-1]), 1)
        pred_fake = self.netD(fake_pair)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.pred_out[-1], self.i2i_label) * 100.0

        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

        return self.loss_G

    def backward_default(self):

        self.aux_loss0 = self.ohem_criterion(self.heads[0](self.pred_out[0]), self.label)
        self.aux_loss1 = self.ohem_criterion(self.heads[1](self.pred_out[1]), self.label)
        self.main_loss = self.ohem_criterion(self.heads[-1](self.pred_out[2]), self.label)

        fake_pair = torch.cat((self.data_resized, self.pred_out[-1]), 1)
        pred_fake = self.netD(fake_pair)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.aux_loss2 = self.ohem_criterion(self.pred_out[-2], self.label)

        self.loss_G = self.loss_G_GAN + self.aux_loss2

        self.loss = self.main_loss + self.aux_loss0 + self.aux_loss1

        self.total_loss = (self.loss * 0.7) + (self.loss_G * 0.3)
        self.total_loss.backward()

        return self.loss, self.loss_G

    def optimize_parameters(self, data, label=None, label_org=None, epoch=None, niters=None, idx=None, lr_policy=None, m=1):

        self.label = label
        self.i2i_label = label_org
        self.forward(data)

        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        lossD = self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights

        self.optimizer_bisenet.zero_grad()
        loss_bisenet, lossG = self.backward_default()
        self.optimizer_bisenet.step()

        current_idx = epoch * niters + idx
        lr = lr_policy.get_lr(current_idx)

        self.optimizer_bisenet.param_groups[0]['lr'] = lr
        self.optimizer_bisenet.param_groups[1]['lr'] = lr / 100
        for i in range(3, len(self.optimizer_bisenet.param_groups)):
            self.optimizer_bisenet.param_groups[i]['lr'] = lr * 10

        return lossD, lossG, loss_bisenet

    def set_default_optimizer(self, default_optimizer=None):

        self.optimizer_bisenet = default_optimizer

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

class BiSeNet(nn.Module):
    def __init__(self, out_planes, is_training,
                 criterion, ohem_criterion, pretrained_model=None,
                 norm_layer=nn.BatchNorm2d):
        super(BiSeNet, self).__init__()
        self.context_path = resnet18(pretrained_model, norm_layer=norm_layer,
                                     bn_eps=config.bn_eps,
                                     bn_momentum=config.bn_momentum,
                                     deep_stem=False, stem_width=64)

        self.business_layer = []
        self.is_training = is_training

        self.spatial_path = SpatialPath(3, 128, norm_layer)

        conv_channel = 128
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(512, conv_channel, 1, 1, 0,
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer)
        )

        arms = [AttentionRefinement(512, conv_channel, norm_layer),
                AttentionRefinement(256, conv_channel, norm_layer)]
        refines = [ConvBnRelu(conv_channel, conv_channel, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, has_bias=False),
                   ConvBnRelu(conv_channel, conv_channel, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, has_bias=False)]

        if is_training:
            heads = [BiSeNetHead(conv_channel, out_planes, 2,
                                 True, norm_layer),
                     BiSeNetHead(conv_channel, out_planes, 1,
                                 True, norm_layer),
                     BiSeNetHead(conv_channel * 2, out_planes, 1,
                                 False, norm_layer)]
        else:
            heads = [None, None,
                     BiSeNetHead(conv_channel * 2, out_planes, 1,
                                 False, norm_layer)]

        self.ffm = FeatureFusion(conv_channel * 2, conv_channel * 2,
                                 1, norm_layer)

        self.arms = nn.ModuleList(arms)
        self.refines = nn.ModuleList(refines)
        self.heads = nn.ModuleList(heads)

        self.business_layer.append(self.spatial_path)
        self.business_layer.append(self.global_context)
        self.business_layer.append(self.arms)
        self.business_layer.append(self.refines)
        self.business_layer.append(self.heads)
        self.business_layer.append(self.ffm)

        if is_training:
            self.criterion = criterion
            self.ohem_criterion = ohem_criterion

    def forward(self, data, label=None):
        spatial_out = self.spatial_path(data)

        context_blocks = self.context_path(data)
        context_blocks.reverse()

        global_context = self.global_context(context_blocks[0])
        global_context = F.interpolate(global_context,
                                       size=context_blocks[0].size()[2:],
                                       mode='bilinear', align_corners=True)

        last_fm = global_context
        pred_out = []

        for i, (fm, arm, refine) in enumerate(zip(context_blocks[:2], self.arms,
                                                  self.refines)):
            fm = arm(fm)
            fm += last_fm
            last_fm = F.interpolate(fm, size=(context_blocks[i + 1].size()[2:]),
                                    mode='bilinear', align_corners=True)
            last_fm = refine(last_fm)
            pred_out.append(last_fm)
        context_out = last_fm

        concate_fm = self.ffm(spatial_out, context_out)
        pred_out.append(concate_fm)

        if self.is_training:
            aux_loss0 = self.ohem_criterion(self.heads[0](pred_out[0]), label)
            aux_loss1 = self.ohem_criterion(self.heads[1](pred_out[1]), label)
            main_loss = self.ohem_criterion(self.heads[-1](pred_out[2]), label)

            loss = main_loss + aux_loss0 + aux_loss1
            return loss

        return F.log_softmax(self.heads[-1](pred_out[-1]), dim=1)


class SpatialPath(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm2d):
        super(SpatialPath, self).__init__()
        inner_channel = 64
        self.conv_7x7 = ConvBnRelu(in_planes, inner_channel, 7, 2, 3,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_3x3_1 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
                                     has_bn=True, norm_layer=norm_layer,
                                     has_relu=True, has_bias=False)
        self.conv_3x3_2 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
                                     has_bn=True, norm_layer=norm_layer,
                                     has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(inner_channel, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)

    def forward(self, x):
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        output = self.conv_1x1(x)

        return output


class BiSeNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d):
        super(BiSeNetHead, self).__init__()
        if is_aux:
            self.conv_3x3 = ConvBnRelu(in_planes, 128, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        else:
            self.conv_3x3 = ConvBnRelu(in_planes, 64, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        # self.dropout = nn.Dropout(0.1)
        if is_aux:
            self.conv_1x1 = nn.Conv2d(128, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        else:
            self.conv_1x1 = nn.Conv2d(64, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        self.scale = scale

    def forward(self, x):
        fm = self.conv_3x3(x)
        # fm = self.dropout(fm)
        output = self.conv_1x1(fm)
        if self.scale > 1:
            output = F.interpolate(output, scale_factor=self.scale,
                                   mode='bilinear',
                                   align_corners=True)

        return output


if __name__ == "__main__":
    model = BiSeNet(19, None)
    # print(model)
