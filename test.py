from model.model import Generator, Discriminator
from utils.datasets import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
from utils.utils import load_model, visual_test
import torch.nn as nn
import torch
from utils.metric import evaluate
from tqdm import tqdm
import numpy
import logging
torch.set_printoptions(threshold=torch.inf)

def test(config, epoch, datalist, log_file, writer):
    transform = T.Compose([T.Resize((512, 512)), T.ToTensor()])
    test_data = Dataset(config, transform, datalist, if_train='test')
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKER)

    generator = Generator(in_channels=config.INPUT_CHANNELS)
    discriminator = Discriminator(in_channels=config.INPUT_CHANNELS, use_sigmoid=True)

    ckpt_dir = config.SAVE_PATH + "dataset_" + str(config.DATASET) + "/subset_" + str(config.SUBSET) + "/anomaly_num_" \
               + str(config.ANOMALY_NUM) + "/seed_" + str(config.SEED) + "/lr_" + str(config.LR) + "/d2glr_" + str(config.D2G_LR)
    load_path = ckpt_dir + "/epoch_" + str(epoch) + '.pth'
    generator, discriminator = load_model(load_path, generator, discriminator, None, None, ifresume=False)
    generator.eval()
    discriminator.eval()
    generator = generator.cuda()
    discriminator = discriminator.cuda()


    select_scale = []
    anomaly_score = []
    gt_label = []
    # cls_label = []
    gt_images = []
    error_map = []
    # count norm on trainset
    print("counting norm...")
    mean_val_error = get_val_error(config, datalist, generator)
    # mean_val_error = torch.zeros(3)
    print("testing...")
    for iter, (masks, test_img, gt_img, label) in tqdm(enumerate(test_loader)):

        error_map_list, concat_horizon, concat_vertical = error_map_for_scale(config, generator, masks, test_img)
        error_map_final, max_scale_ind = get_max_error_map(config, error_map_list, mean_val_error)
        # if epoch % 5 == 0 and (iter + 1) % config.VIS_INTERVAL == 0:
        #     visual_test(config, epoch, iter, test_img, concat_horizon, concat_vertical, error_map_final)

        select_scale += max_scale_ind
        anomaly_score += torch.mean(error_map_final, [1, 2])
        gt_label += label
        # cls_label += cls
        gt_img = torch.mean(gt_img, dim=1)
        gt_images += gt_img.to(torch.int)
        error_map += error_map_final
    # select_scale = torch.tensor(select_scale)
    anomaly_score = torch.tensor(anomaly_score)
    gt_label = torch.tensor(gt_label)
    # cls_label = torch.tensor(cls_label)
    gt_images = torch.stack(gt_images).flatten(0)   # N/bs, bs, 512, 512 -> N, 512, 512
    error_map = torch.stack(error_map).flatten(0)
    # gt_images = torch.tensor(gt_images).view(-1, 512, 512)  # N/bs, bs, 512, 512 -> N, 512, 512
    # error_map = torch.tensor(error_map).view(-1, 512, 512)

    anomaly_score = (anomaly_score - torch.min(anomaly_score)) / (torch.max(anomaly_score) - torch.min(anomaly_score))
    log_file.write("Testing...\n")
    # for cls, cls_idx in cls_name.items():
    #     gt_cls = []
    #     score_cls = []
    #     error_map_cls = []
    #     gt_images_cls = []
    #     for i, cl in enumerate(cls_label):
    #         if cl == cls_idx:
    #             gt_cls.append(gt_label[i])
    #             score_cls.append(anomaly_score[i])
    #             error_map_cls.append(error_map[i])
    #             gt_images_cls.append(gt_images[i])
    #     gt_cls = torch.tensor(gt_cls)
    #     score_cls = torch.tensor(score_cls)
    #     error_map_cls = torch.stack(error_map_cls).flatten(0)
    #     gt_images_cls = torch.stack(gt_images_cls).flatten(0)
    if config.DATASET =='mvtec':
        cls_name = config.SUBSET
    # 对每个类别计算检测判断是否为异常的auc
    auc_ad = evaluate((1 - gt_label), anomaly_score)
    # 对每个类别计算gt和预测的errormap之间每个像素位auc平均
    auc_gt = evaluate(gt_images, error_map)
    # 保存最佳结果

    log = {'detection_auc': auc_ad, 'region_auc': auc_gt}
    log_file.write(str(log) + '\n')
    print(cls_name, 'auc:%.8f' % auc_ad, 'region auc:%.8f' % auc_gt)

    dict_cls = {'detection_auc': auc_ad, 'region_auc': auc_gt}
    writer.add_scalars(main_tag=cls_name, tag_scalar_dict=dict_cls, global_step=epoch)

    return auc_ad, auc_gt


def error_map_for_scale(config, generator, masks, test_img):
    """
    计算每个尺度的error map
    :param config:
    :param generator:
    :param masks:
    :param test_img:
    :return: error_map_list (3, bs, 512, 512) 3个scale上的error map
             concat_horizon, concat_vertical (bs, 3, 512, 512)x3 重构后拼贴起来的生成图
    """
    mse = nn.MSELoss(reduction='none')
    test_img = test_img.cuda()
    with torch.no_grad():
        error_map_list = []
        concat_horizon = []
        concat_vertical = []
        for i in range(len(config.SCALES)):
            mask_output = []
            error_map = []
            for mask in masks[i * 4:(i + 1) * 4]:
                mask = mask.cuda()
                masked_img = mask * test_img
                outputs = generator(masked_img)
                mask_output.append(outputs * (1 - mask))
                error_map.append(torch.mean(mse(outputs, test_img) * (1 - mask), 1))  # 3通道求平均，这一维会压掉
            mask_output = torch.stack(mask_output)
            concat_horizon.append(torch.sum(mask_output[0:2], 0))  # 每个scale横竖各拼接一张
            concat_vertical.append(torch.sum(mask_output[2:4], 0))
            error_map = torch.stack(error_map)  # 将list或tuple合并起来，增加第0维，变为tensor方便每个位置上求max
            error_map_list.append(torch.max(error_map, 0)[0])
            # 四种mask拼到第0维，对每个位置上取0维最大值，然后[0]相当于squeeze，把选出的最大的0维去掉得到(1,512,512)
    return error_map_list, concat_horizon, concat_vertical

def get_val_error(config, datalist, generator):
    transform = T.Compose([T.Resize((512, 512)), T.ToTensor()])
    val_data = Dataset(config, transform, datalist, if_train='val')
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKER)
    mean_val_error = [0] * len(config.SCALES)
    for index, (masks, images) in tqdm(enumerate(val_loader)):
        val_error_map, concat1, concat2 = error_map_for_scale(config, generator, masks, images)
        for i, scale in enumerate(config.SCALES):
            mean_val_error[i] += torch.mean(val_error_map[i]) * len(images) / len(val_data)     # batch上平均error * bs /N
    return mean_val_error

def get_max_error_map(config, error_map_list, mean_val_error):
    """
    :param config:
    :param error_map_list: (3, bs, 512, 512) 3个scale上的error map
    :param mean_val_error:
    :return: error_map_final (bs, 512, 512), max_scale (bs)
    """
    error_map_list = torch.stack(error_map_list)
    mean_test_error = torch.mean(error_map_list, [2, 3])    # [3, bs]
    diff_on_scale = torch.zeros(mean_test_error.size())
    for i, scale in enumerate(config.SCALES):
        diff_on_scale[i] = mean_test_error[i] - mean_val_error[i]   # [3, bs] = [3, bs] - [3]
    diff_on_scale, max_scale = torch.max(diff_on_scale, dim=0)   # [bs]
    error_map_final = torch.stack([error_map_list[scale, bs] for bs, scale in enumerate(max_scale)])  # [bs] batch上对每张图片上选一个最大的scale
    return error_map_final, max_scale







