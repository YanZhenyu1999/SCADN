from model.model import Generator, Discriminator
from utils.datasets import Datalist, Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn as nn
from torch.optim import Adam
import torch
import numpy as np
import os
import yaml
import easydict
import argparse
import logging
import shutil
from test import test
from tensorboardX import SummaryWriter
from pathlib import Path
from utils.utils import save_model, load_model, visual_train

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, nargs='+', help='gpu list')
parser.add_argument('--config', type=str, help='the path of config file', default='./configs/config_mvtec.yml')
parser.add_argument('--checkpoint', type=str, help='model checkpoints path for continuing training', default=None)
parser.add_argument('--subset', type=str, help='the category of dataset to train', default=None)
parser.add_argument('--train_anomaly', type=int,  default=None, help='anomaly images num to train')
# parser.add_argument('--train_anomaly', default=False, action='store_true', help='if use anomaly images to train')
args = parser.parse_args()

config = yaml.safe_load(open(args.config))
config = easydict.EasyDict(config)

if args.subset is not None:
    config.SUBSET = args.subset

if args.gpu is not None:
    config.GPU = list(args.gpu)
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(id) for id in config.GPU)

if args.train_anomaly is not None:
    config.ANOMALY_NUM = args.train_anomaly
else:
    config.ANOMALY_NUM = 0


# 日志保存路径
logname = "dataset_" + str(config.DATASET) + "/subset_" + str(config.SUBSET) + "/anomaly_num_" + str(config.ANOMALY_NUM) + "_seed_" + str(config.SEED) + "_lr_" + str(config.LR) + "_d2glr_" + str(config.D2G_LR) + ".log"
logpath = os.path.join(config.LOG_PATH, logname)
if not os.path.exists(os.path.dirname(logpath)):
    os.makedirs(os.path.dirname(logpath))
# logger = logging.getLogger()
# logging.basicConfig(filename=logname, format="%(message)s", filemode='a')
# logger.setLevel(logging.INFO)

write_dir = config.WRITE_PATH + "dataset_" + str(config.DATASET) + "_subset_" + str(config.SUBSET) + "/anomaly_num_" + str(config.ANOMALY_NUM) + '_seed_' + str(config.SEED) + "_lr_" + str(config.LR) + "_d2glr_" + str(config.D2G_LR)
if not os.path.exists(write_dir):
    os.makedirs(write_dir)
elif args.checkpoint is None:
    shutil.rmtree(write_dir)
    os.mkdir(write_dir)
writer = SummaryWriter(write_dir)

# checkpoint保存路径
ckpt_dir = config.SAVE_PATH + "dataset_" + str(config.DATASET) + "/subset_" + str(config.SUBSET) + "/anomaly_num_" + str(config.ANOMALY_NUM) + "/seed_" + str(config.SEED) + "/lr_" + str(config.LR) + "/d2glr_" + str(config.D2G_LR)
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

transform = T.Compose([T.Resize((512, 512)), T.RandomHorizontalFlip(p=0.5), T.ToTensor()])
datalist = Datalist(config, train_anomaly=args.train_anomaly)
train_data = Dataset(config, transform, datalist, if_train='train')
train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKER, shuffle=True)

generator = Generator(in_channels=config.INPUT_CHANNELS).cuda()
discriminator = Discriminator(in_channels=config.INPUT_CHANNELS, use_sigmoid=True).cuda()
if torch.cuda.device_count() > 1:
    generator = nn.DataParallel(generator, device_ids=range(torch.cuda.device_count() - 1))
    discriminator = nn.DataParallel(discriminator, device_ids=range(torch.cuda.device_count() - 1))
mse = nn.MSELoss(reduction='none')
bce = nn.BCELoss(reduction='none')
optimG = Adam(lr=config.LR, betas=(config.BETA1, config.BETA2), params=generator.parameters())
optimD = Adam(lr=config.LR * config.D2G_LR, betas=(config.BETA1, config.BETA2), params=discriminator.parameters())

if args.checkpoint is not None:
    generator, discriminator, optimG, optimD, start_epoch,\
    auc_ad_best, auc_gt, best_ad_epoch, auc_ad, auc_gt_best, best_gt_epoch = load_model(args.checkpoint, generator, discriminator, optimG, optimD, ifresume=True)
else:
    start_epoch = 1
    auc_ad_best = auc_gt_best = auc_ad = auc_gt = best_ad_epoch = best_gt_epoch = 0

log_ad_best = {'best_detection_auc': auc_ad_best, 'region_auc': auc_gt, 'best_ad_epoch': best_ad_epoch}
log_gt_best = {'detection_auc': auc_ad, 'best_region_auc': auc_gt_best, 'best_gt_epoch': best_gt_epoch}

for epoch in range(start_epoch, config.EPOCHS + 1):
    log_file = open(logpath, 'a')
    for iter, (mask, images, gt, label) in enumerate(train_loader):

        generator.train()
        discriminator.train()

        # print(masks.size(), images.size())
        images = images.cuda()
        mask = mask.cuda()
        label = label.cuda()
        gt = gt.cuda()
        masked_images = mask * images
        masked_images = masked_images.cuda()


        # train D
        # compute loss of real images, label is 1, abnormaly is 0
        dis_loss = 0
        optimD.zero_grad()
        real_output, real_feat = discriminator(images)
        # print(real_output.type()) # bs,1,62,62
        # real_label = label.view(len(label), 1, 1, 1)
        # real_label = real_label.expand_as(real_output)
        # real_label = torch.as_tensor(real_label, dtype=torch.float32).cuda()
        real_loss = bce(real_output, gt)
        real_loss = real_loss.mean()

        # compute loss of fake images, label is 0
        outputs = generator(masked_images)
        fake = outputs.detach()     # 计算discriminator的loss时，梯度不回传给generator
        fake_output, fake_feat = discriminator(fake)
        fake_label = torch.zeros(fake_output.size()).cuda()
        fake_loss = torch.mean(bce(fake_output, fake_label), dim=[1, 2, 3]) * label   # 在除bs以外做平均，乘以label权重，不需要anomaly的梯度
        fake_loss = fake_loss.mean()

        dis_loss += (real_loss + fake_loss) / 2
        dis_loss.backward()
        optimD.step()

        # train G
        # generator adversarial loss
        gen_loss = 0
        optimG.zero_grad()
        gen_fake, gen_fake_feat = discriminator(outputs)
        # print(gen_fake.size())
        # print(real_label.size())
        # gen_adv_loss = bce(gen_fake, real_label)
        # print(gen_adv_loss.size())
        gen_adv_loss = torch.mean(bce(gen_fake, gt), dim=[1, 2, 3]) * label
        gen_adv_loss = gen_adv_loss.mean()

        # generator reconstruction loss
        rec_unmask = torch.mean(mse(outputs * (1 - mask), images * (1 - mask)), [1, 2, 3]) * label
        rec_mask = torch.mean(mse(outputs * mask, images * mask), [1, 2, 3]) * label
        rec_loss = (rec_unmask.mean() + 4 * rec_mask.mean()) / torch.mean(mask)
        gen_rec_loss = rec_loss
        gen_loss += config.ADV_LOSS_WEIGHT * gen_adv_loss + config.REC_LOSS_WEIGHT * gen_rec_loss

        # generator feature matching loss
        # gen_fm_loss = 0
        # for i in range(len(real_feat)):
        #     gen_fm_loss += mse(gen_fake_feat[i], real_feat[i].detach())
        # gen_fm_loss = gen_fm_loss * config.FM_LOSS_WEIGHT
        # gen_loss += gen_fm_loss

        gen_loss.backward()
        optimG.step()

        # dictionary
        # logs = {
        #     "epoch": epoch,
        #     'iter': iter + 1,
        #     "l_dis": dis_loss.item(),
        #     "l_dis_real": real_loss.item(),
        #     "l_dis_fake": fake_loss.item(),
        #     "l_gen_gan": gen_adv_loss.item(),
        #     "l_l1": gen_rec_loss.item(),
        #     # "l_fm": gen_fm_loss.item(),
        #     "l_gen_sum": gen_loss.item(),
        # }

        # string
        logs = 'Epoch[{}/{}] iter[{}], '.format(epoch, config.EPOCHS, iter + 1) + \
               'l_dis: {:.8f}, '.format(dis_loss.item()) + \
               'l_gen: {:.8f}, '.format(gen_loss.item()) + \
               'l_dis_real: {:.8f}, '.format(real_loss.item()) + \
               'l_dis_fake: {:.8f}, '.format(fake_loss.item()) + \
               'l_gen_adv: {:.8f}, '.format(gen_adv_loss.item()) + \
               'l_gen_rec: {:.8f}'.format(gen_rec_loss.item())


        if (iter + 1) % config.LOG_INTERVAL == 0:
            print(logs)
            log_file.write(str(logs) + '\n')
            writer.add_scalar("dis_loss", dis_loss.item(), iter)
            writer.add_scalar("gen_loss", gen_loss.item(), iter)
            dict_loss = {"dis_loss": dis_loss.item(), "gen_loss": gen_loss.item()}
            writer.add_scalars(main_tag="loss", tag_scalar_dict=dict_loss, global_step=(epoch-1)*len(train_loader)+iter)

        # if (iter + 1) % config.VIS_INTERVAL == 0:
        #     visual_train(config, images, masked_images, outputs, epoch, iter)

    if epoch % config.CKPT_INTERVAL == 0:
        ckpt_path = ckpt_dir + "/epoch_" + str(epoch) + '.pth'
        save_model(ckpt_path, generator, discriminator, optimG, optimD, epoch, log_ad_best, log_gt_best)

    if epoch % config.EVAL_INTERVAL == 0:
        auc_ad, auc_gt = test(config, epoch, datalist, log_file, writer)
        if auc_ad > auc_ad_best:
            auc_ad_best = auc_ad
            best_ad_epoch = epoch
            log_ad_best = {'best_detection_auc': auc_ad_best, 'region_auc': auc_gt, 'best_ad_epoch': best_ad_epoch}
        if auc_gt > auc_gt_best:
            auc_gt_best = auc_gt
            best_gt_epoch = epoch
            log_gt_best = {'detection_auc': auc_ad, 'best_region_auc': auc_gt_best, 'best_gt_epoch': best_gt_epoch}

    log_file.close()
    writer.close()

log_file = open(logpath, 'a')
log_file.write("best detection auc\n")
log_file.write(str(log_ad_best) + '\n')
log_file.write("best region auc\n")
log_file.write(str(log_gt_best) + '\n')
log_file.close()

print(log_ad_best)
print(log_gt_best)