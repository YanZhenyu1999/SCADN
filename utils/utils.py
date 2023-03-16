import os.path
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl
import torch.nn as nn

# mpl.rcParams['font.sans-serif'] = ['FangSong']

def save_model(save_path, model_g, model_d, optim_g, optim_d, epoch, log_ad_best, log_gt_best):
    save_dic = {
        'model_g': model_g.state_dict(),
        'model_d': model_d.state_dict(),
        'optim_g': optim_g.state_dict(),
        'optim_d': optim_d.state_dict(),
        'epoch': epoch,
        'best_detection_auc': log_ad_best['best_detection_auc'],
        'region_auc': log_ad_best['region_auc'],
        'best_ad_epoch': log_ad_best['best_ad_epoch'],
        'detection_auc': log_gt_best['detection_auc'],
        'best_region_auc': log_gt_best['best_region_auc'],
        'best_gt_epoch': log_gt_best['best_gt_epoch']
    }
    torch.save(save_dic, save_path)


def load_model(load_path, model_g, model_d, optim_g, optim_d, ifresume):
    checkpoint = torch.load(load_path)
    if ifresume:
        model_g.load_state_dict(checkpoint['model_g'])
        model_d.load_state_dict(checkpoint['model_d'])
        optim_g.load_state_dict(checkpoint['optim_g'])
        optim_d.load_state_dict(checkpoint['optim_d'])
        start_epoch = checkpoint['epoch']
        auc_ad_best = checkpoint['best_detection_auc']
        auc_gt = checkpoint['region_auc']
        best_ad_epoch = checkpoint['best_ad_epoch']
        auc_ad = checkpoint['detection_auc']
        auc_gt_best = checkpoint['best_region_auc']
        best_gt_epoch = checkpoint['best_gt_epoch']
        return model_g, model_d, optim_g, optim_d, start_epoch, auc_ad_best, auc_gt, best_ad_epoch, auc_ad, auc_gt_best, best_gt_epoch
    else:
        g_dict = checkpoint['model_g']
        d_dict = checkpoint['model_d']
        if torch.cuda.device_count() > 1:
            model_g = nn.DataParallel(model_g, device_ids=range(torch.cuda.device_count() - 1))
            model_d = nn.DataParallel(model_d, device_ids=range(torch.cuda.device_count() - 1))
        # for k, v in param_g.items():
        #     if 'module.' in k:
        #         g_dict[k.replace('module.', '')] = v
        # for k, v in param_d.items():
        #     if 'module.' in k:
        #         d_dict[k.replace('module.', '')] = v
        model_g.load_state_dict(g_dict)
        model_d.load_state_dict(d_dict)
        return model_g, model_d


def visual_train(config, img, masked, generation, epochs, iter):
    """
    :param img: batch of the input images (tensor)
    :param masked: batch of the masked images (tensor)
    :param generation: batch of the generated images (tensor)
    :param epochs:
    :param iter:
    :return:
    """
    path = config.VIS_TRAIN + config.DATASET
    if config.DATASET == 'mvtec':
        path = path + "/" + config.SUBSET + "/"
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    bs = img.size(0)
    img, masked, generation = img.detach().cpu(), masked.detach().cpu(), generation.detach().cpu()
    for i in range(bs):
        original_img = img[i, :, :, :].permute(1, 2, 0).numpy()
        masked_img = masked[i, :, :, :].permute(1, 2, 0).numpy()
        generated_img = generation[i, :, :, :].permute(1, 2, 0).numpy()
        plt.subplot(bs, 3, i * 3 + 1)
        plt.imshow(original_img)
        plt.subplot(bs, 3, i * 3 + 2)
        plt.imshow(masked_img)
        plt.subplot(bs, 3, i * 3 + 3)
        plt.imshow(generated_img)

    plt.axis('off')
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    # print(str(path) + str(epochs) + '_' + str(iter + 1) + '.png')
    plt.savefig(str(path) + str(epochs) + '_' + str(iter + 1) + '.png')


def visual_test(config, epoch, iter, test, concat1, concat2, errormap):
    """
    :param config:
    :param test: (bs, 3, 512, 512)
    :param concat1: 长度为len(SCALES)的列表，列表内元素为(bs, 3, 512, 512)的tensor
    :param concat2:
    :param iter:
    :return:
    """
    path = config.VIS_TEST + config.DATASET
    if config.DATASET == 'mvtec':
        path = path + "/" + config.SUBSET + "/"
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    # k = len(config.SCALES)
    bs = config.BATCH_SIZE
    # print(len(concat1), concat1[0].size())
    test = test.detach().cpu()
    h1 = concat1[0].detach().cpu()  # k = 4
    h2 = concat1[1].detach().cpu()  # k = 8
    h3 = concat1[2].detach().cpu()  # k = 16
    v1 = concat2[0].detach().cpu()
    v2 = concat2[1].detach().cpu()
    v3 = concat2[2].detach().cpu()
    errormap = errormap.detach().cpu()
    for i in range(bs):
        test_img = test[i, :, :, :].permute(1, 2, 0).numpy()
        h1_img = h1[i, :, :, :].permute(1, 2, 0).numpy()
        h2_img = h2[i, :, :, :].permute(1, 2, 0).numpy()
        h3_img = h3[i, :, :, :].permute(1, 2, 0).numpy()
        error_img = errormap[i, :, :].numpy()
        f1 = plt.subplot(4, 5, i * 5 + 1)
        f1.set_title("test image")
        plt.imshow(test_img)
        f2 = plt.subplot(4, 5, i * 5 + 2)
        f2.set_title("scale=4")
        plt.imshow(h1_img)
        f3 = plt.subplot(4, 5, i * 5 + 3)
        f3.set_title("scale=8")
        plt.imshow(h2_img)
        f4 = plt.subplot(4, 5, i * 5 + 4)
        f4.set_title("scale=16")
        plt.imshow(h3_img)
        f5 = plt.subplot(4, 5, i * 5 + 5)
        f5.set_title("error map")
        plt.imshow(error_img)
    plt.axis('off')
    # print(path)
    plt.savefig(path + 'h_' + str(epoch) + '_' + str(iter + 1) + '.png')

    for j in range(bs):
        test_img = test[i, :, :, :].permute(1, 2, 0).numpy()
        v1_img = v1[j, :, :, :].permute(1, 2, 0).numpy()
        v2_img = v2[j, :, :, :].permute(1, 2, 0).numpy()
        v3_img = v3[j, :, :, :].permute(1, 2, 0).numpy()
        error_img = errormap[i, :, :].numpy()
        f1 = plt.subplot(4, 5, j * 5 + 1)
        f1.set_title("test image")
        plt.imshow(test_img)
        f2 = plt.subplot(4, 5, j * 5 + 2)
        f2.set_title("scale=4")
        plt.imshow(v1_img)
        f3 = plt.subplot(4, 5, j * 5 + 3)
        f3.set_title("scale=8")
        plt.imshow(v2_img)
        f4 = plt.subplot(4, 5, j * 5 + 4)
        f4.set_title("scale=16")
        plt.imshow(v3_img)
        f5 = plt.subplot(4, 5, j * 5 + 5)
        f5.set_title("error map")
        plt.imshow(error_img)
    plt.axis('off')
    plt.savefig(path + 'v_' + str(epoch) + '_' + str(iter + 1) + '.png')
