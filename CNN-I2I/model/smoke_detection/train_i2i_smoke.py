from __future__ import division
import sys
import argparse
from tqdm import tqdm
import os.path as osp
import os

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image

from config import config
from dataloader import get_train_loader_Smoke
from network import BiSeNet_smoke
from datasets.smoke import Smoke as IDataset
from utils.init_func import init_weight, group_weight
from engine.lr_policy import PolyLR
from engine.engine import Engine
from seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d

parser = argparse.ArgumentParser()

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True

    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    train_loader, train_sampler = get_train_loader_Smoke(engine, IDataset)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean',
                                    ignore_index=255)

    ohem_criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7,
                                            min_kept=int(
                                                config.batch_size // len(
                                                    engine.devices) * config.image_height * config.image_width // (
                                                        16 * config.gt_down_sampling ** 2)),
                                            use_weight=False)

    if engine.distributed:
        BatchNorm2d = SyncBatchNorm

    BatchNorm2d = torch.nn.BatchNorm2d

    model = BiSeNet_smoke(config.num_classes, is_training=True,
                    criterion=criterion,
                    ohem_criterion=ohem_criterion,
                    pretrained_model=config.pretrained_model,
                    norm_layer=BatchNorm2d)

    init_weight(model.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    # group weight and config optimizer
    base_lr = config.lr

    params_list = []
    params_list = group_weight(params_list, model.context_path,
                               BatchNorm2d, base_lr)
    params_list = group_weight(params_list, model.i2i_path,
                               BatchNorm2d, base_lr / 100)
    params_list = group_weight(params_list, model.spatial_path,
                               BatchNorm2d, base_lr * 10)
    params_list = group_weight(params_list, model.global_context,
                               BatchNorm2d, base_lr * 10)
    params_list = group_weight(params_list, model.arms,
                               BatchNorm2d, base_lr * 10)
    params_list = group_weight(params_list, model.i2i_arms,
                               BatchNorm2d, base_lr * 10)
    params_list = group_weight(params_list, model.i2i_ffms,
                               BatchNorm2d, base_lr * 10)
    params_list = group_weight(params_list, model.refines,
                               BatchNorm2d, base_lr * 10)
    params_list = group_weight(params_list, model.heads,
                               BatchNorm2d, base_lr * 10)
    params_list = group_weight(params_list, model.ffm,
                               BatchNorm2d, base_lr * 10)

    optimizer = torch.optim.SGD(params_list,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    model.set_default_optimizer(optimizer)

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = PolyLR(base_lr, config.lr_power, total_iteration)

    if engine.distributed:
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model)
    else:
        device = torch.device(config.gpu_number if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    model.train()

    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)
        for idx in pbar:
            engine.update_iteration(epoch, idx)

            minibatch = next(dataloader)
            imgs = minibatch['data']
            gts = minibatch['label']
            gts_org = minibatch['i2i_label']



            imgs = imgs.to(device)
            gts = gts.to(device)
            gts_org = gts_org.to(device)

            loss_D, loss_G, loss_BiSeNet = model.optimize_parameters(data=imgs, label=gts, label_org=gts_org,
                                                                           epoch=epoch,
                                                                           niters=config.niters_per_epoch,
                                                                           idx=idx,
                                                                           lr_policy=lr_policy)

            if engine.distributed:
                dist.all_reduce(loss, dist.ReduceOp.SUM)
                loss = loss / engine.world_size

            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            if (idx+1) % 100 == 0:
                i2i_pred = model.pred_out[-1]
                i2i_pred = i2i_pred[0, :, :, :]
                save_gt = gts_org[0, :, :, :]

                downsampling_rate_height = config.image_height // config.gt_down_sampling # change
                downsampling_rate_width = config.image_width // config.gt_down_sampling # cahnge

                mean_tensor_R = torch.full([1, downsampling_rate_height, downsampling_rate_width], config.image_mean[0], device=device) # change
                mean_tensor_G = torch.full([1, downsampling_rate_height, downsampling_rate_width], config.image_mean[1], device=device) # change
                mean_tensor_B = torch.full([1, downsampling_rate_height, downsampling_rate_width], config.image_mean[2], device=device) # change
                mean_tensor_RGB = torch.cat((mean_tensor_R, mean_tensor_G, mean_tensor_B), 0)
                std_tensor_R = torch.full([1, downsampling_rate_height, downsampling_rate_width], config.image_std[0], device=device)  # change
                std_tensor_G = torch.full([1, downsampling_rate_height, downsampling_rate_width], config.image_std[1], device=device)  # change
                std_tensor_B = torch.full([1, downsampling_rate_height, downsampling_rate_width], config.image_std[2], device=device)  # change
                std_tensor_RGB = torch.cat((std_tensor_R, std_tensor_G, std_tensor_B), 0) # change

                mean_tensor_i2i_R = torch.full([1, downsampling_rate_height, downsampling_rate_width], 0.5, device=device) # change
                mean_tensor_i2i_G = torch.full([1, downsampling_rate_height, downsampling_rate_width], 0.5, device=device) # change
                mean_tensor_i2i_B = torch.full([1, downsampling_rate_height, downsampling_rate_width], 0.5, device=device) # change
                mean_tensor_i2i_RGB = torch.cat((mean_tensor_i2i_R, mean_tensor_i2i_G, mean_tensor_i2i_B), 0)
                std_tensor_i2i_R = torch.full([1, downsampling_rate_height, downsampling_rate_width], 0.5, device=device) # change
                std_tensor_i2i_G = torch.full([1, downsampling_rate_height, downsampling_rate_width], 0.5, device=device) # change
                std_tensor_i2i_B = torch.full([1, downsampling_rate_height, downsampling_rate_width], 0.5, device=device) # change
                std_tensor_i2i_RGB = torch.cat((std_tensor_i2i_R, std_tensor_i2i_G, std_tensor_i2i_B), 0)

                save_org_img = model.data_resized[0, 0:3, :, :] * std_tensor_RGB + mean_tensor_RGB # change
                save_gt_img = save_gt * std_tensor_i2i_RGB + mean_tensor_i2i_RGB
                save_i2i_pred_img = i2i_pred * std_tensor_i2i_RGB + mean_tensor_i2i_RGB
                save_img = torch.cat((save_org_img, save_gt_img, save_i2i_pred_img), 2)
                filename = osp.abspath(osp.join(config.Imgs_dir, 'img{}_{}.jpg'.format(epoch, idx+1))) # change
                if not osp.exists(config.Imgs_dir): # change
                    os.makedirs(config.Imgs_dir) # change

                save_image(save_img, filename)


            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss_BiSeNet=%.3f' % loss_BiSeNet.item() \
                        + ' loss_D=%.3f' % loss_D.item() \
                        + ' loss_G=%.3f' % loss_G.item()


            pbar.set_description(print_str, refresh=False)

        if (epoch > config.nepochs - 100) or (epoch % config.snapshot_iter == 0):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
