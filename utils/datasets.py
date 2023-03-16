import random
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import pandas as pd
import numpy as np
import os

class Datalist(object):
    def __init__(self, config, train_anomaly):
        self.dataset = config.DATASET
        self.img_root = config.DATA_ROOT
        self.image_size = config.IMAGE_SIZE
        self.train_path = []
        self.test_path = []
        self.val_path = []
        self.test_gt = []
        self.train_label = []
        self.test_label = []
        self.train_gt = []

        if self.dataset == 'mvtec':
            self.subset = config.SUBSET
            # trainset
            cls_root = os.path.join(self.img_root, self.subset)
            if os.path.isdir(cls_root):
                train_img_root = cls_root + '/train/good/'
                self.train_path += [os.path.join(train_img_root, i) for i in os.listdir(train_img_root)]
                self.train_label += [1] * len(os.listdir(train_img_root))
                self.val_path = self.train_path
                self.train_gt += [self.img_root + '/black.png'] * len(os.listdir(train_img_root))
            # testset
                test_root = cls_root + '/test/'
                for ad in os.listdir(test_root):
                    test_img_root = os.path.join(test_root, ad)
                    if ad == 'good':
                        self.test_path += [os.path.join(test_img_root, i) for i in os.listdir(test_img_root)]
                        self.test_label += [1] * len(os.listdir(test_img_root))
                        self.test_gt += [self.img_root + '/black.png'] * len(os.listdir(test_img_root))
                    else:
                        for ad_img in os.listdir(test_img_root):
                            self.test_path.append(os.path.join(test_img_root, ad_img))
                            self.test_label.append(0)
                            self.test_gt.append(cls_root + '/ground_truth/' + ad + '/' + ad_img.split('.')[0] + '_mask.png')

        if not train_anomaly == 0:
            self.anomaly_for_train = []
            self.anomaly_num = 0
            while self.anomaly_num < config.ANOMALY_NUM:
                random_index = random.randrange(len(self.test_path))
                if self.test_label[random_index] == 0:
                    self.train_path.append(self.test_path[random_index])
                    self.train_label.append(self.test_label[random_index])
                    self.train_gt.append(self.test_gt[random_index])
                    self.test_path.pop(random_index)
                    self.test_label.pop(random_index)
                    self.test_gt.pop(random_index)
                    self.anomaly_num += 1

    def get_val_set(self):
        return self.val_path

    def get_train_set(self):
        return self.train_path, self.train_label, self.train_gt

    def get_test_set(self):
        return self.test_path, self.test_label, self.test_gt


class Dataset(object):
    def __init__(self, config, transform, datalist, if_train):
        self.transform = transform
        self.image_size = config.IMAGE_SIZE
        self.masks = maskset(self.image_size, config.SCALES)
        self.if_train = if_train
        self.val_path = datalist.get_val_set()
        self.train_path, self.train_label, self.train_gt = datalist.get_train_set()
        self.test_path, self.test_label, self.test_gt = datalist.get_test_set()

    def __getitem__(self, item):
        if self.if_train == 'train':
            train = self.train_path[item]
            label = self.train_label[item]
            gt = self.train_gt[item]
            img = Image.open(train).convert('RGB')
            img = self.transform(img)
            gt_img = Image.open(gt).convert('1')
            gt_img = transforms.Compose([transforms.Resize((62, 62)), transforms.ToTensor()])(gt_img)
            mask = random.choice(self.masks)
            return mask, img, 1 - gt_img, label
        elif self.if_train == 'val':
            val = self.val_path[item]
            img = Image.open(val).convert('RGB')
            img = self.transform(img)
            masks = self.masks
            return masks, img
        elif self.if_train == 'test':
            test = self.test_path[item]
            gt = self.test_gt[item]
            label = self.test_label[item]
            test_img = Image.open(test).convert('RGB')
            gt_img = Image.open(gt).convert('RGB')
            gt_img = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])(gt_img)
            test_img = self.transform(test_img)
            masks = self.masks
            return masks, test_img, gt_img, label

    def __len__(self):
        if self.if_train == 'train':
            return len(self.train_path)
        elif self.if_train == 'val':
            return len(self.val_path)
        elif self.if_train == 'test':
            return len(self.test_path)


def maskset(image_size, scales):
    maskset = []
    for scale in scales:
        m = torch.ones((3, image_size, image_size))
        pad = image_size // scale
        for i in range(scale):
            if i % 2 == 0:
                m[:, :, pad * i: pad * (i + 1)] = 0
        maskset += [m, 1-m, torch.transpose(m, 1, 2), torch.transpose(1 - m, 1, 2)]
    return maskset



# class Testset(object):
#     def __init__(self, config, transform):
#         super(Testset, self).__init__()
#         self.dataset = config.DATASET
#         self.img_root = config.DATA_ROOT
#         self.transform = transform
#         self.image_size = config.IMAGE_SIZE
#         self.mask = maskset(self.image_size, config.SCALES)
#         self.test_path = []
#         self.ground_truth = []
#         self.label = []
#         self.cls = []
#         self.cls_name = dict()
#         cls_idx = 0
#
#         # if self.dataset == 'mvtec':
#         #     self.subset = config.SUBSET
#         #     cls_root = os.path.join(self.img_root, self.subset)
#         #     if os.path.isdir(cls_root):
#         #         test_root = cls_root + '/test/'
#         #         for ad in os.listdir(test_root):
#         #             test_img_root = os.path.join(test_root, ad)
#         #             if ad == 'good':
#         #                 self.test_path += [os.path.join(test_img_root, i) for i in os.listdir(test_img_root)]
#         #                 self.label += [0] * len(os.listdir(test_img_root))
#         #                 self.ground_truth += [self.img_root + '/black.png'] * len(os.listdir(test_img_root))
#         #             else:
#         #                 for ad_img in os.listdir(test_img_root):
#         #                     self.test_path.append(os.path.join(test_img_root, ad_img))
#         #                     self.label.append(1)
#         #                     self.ground_truth.append(cls_root + '/ground_truth/' + ad + '/' + ad_img.split('.')[0] + '_mask.png')
#
#         # for cls in os.listdir(self.img_root):
#         #     cls_root = os.path.join(self.img_root, cls)
#         #     if os.path.isdir(cls_root):
#         #         self.cls_name[str(cls)] = cls_idx
#         #         test_root = cls_root + '/test/'
#         #         for ad in os.listdir(test_root):
#         #             test_img_root = os.path.join(test_root, ad)
#         #             if ad == 'good':
#         #                 self.test_path += [os.path.join(test_img_root, i) for i in os.listdir(test_img_root)]
#         #                 self.label += [0] * len(os.listdir(test_img_root))
#         #                 self.cls += [cls_idx] * len(os.listdir(test_img_root))
#         #                 self.ground_truth += [self.img_root + '/black.png'] * len(os.listdir(test_img_root))
#         #             else:
#         #                 for ad_img in os.listdir(test_img_root):
#         #                     self.test_path.append(os.path.join(test_img_root, ad_img))
#         #                     self.label.append(1)
#         #                     self.cls.append(cls_idx)
#         #                     self.ground_truth.append(cls_root + '/ground_truth/' + ad + '/' + ad_img.split('.')[0] + '_mask.png')
#         #         cls_idx += 1
#
#     def __getitem__(self, index):
#         test = self.test_path[index]
#         gt = self.ground_truth[index]
#         label = self.label[index]
#         # cls = self.cls[index]
#         test_img = Image.open(test).convert('RGB')
#         gt_img = Image.open(gt).convert('RGB')
#         gt_img = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])(gt_img)
#         if self.transform is not None:
#             test_img = self.transform(test_img)
#         masks = self.mask
#         return masks, test_img, gt_img, label
#
#     def __len__(self):
#         return len(self.test_path)
#
#
#
#
# class Trainset(object):
#     def __init__(self, config, transform, if_val, if_train_anomaly):
#         """
#         :param img_root: the path of dataset
#         :param transform: image augmentation
#         :param subset: dataset category
#         :param image_size: the size of input images
#         :param if_val: value set for test
#         """
#         super().__init__()
#         self.dataset = config.DATASET
#         self.img_root = config.DATA_ROOT
#         self.transform = transform
#         self.image_size = config.IMAGE_SIZE
#         self.scales = config.SCALES
#         self.anomaly_num = config.ANOMALY_NUM
#         self.mask = maskset(self.image_size, self.scales)
#         self.train_path = []
#         self.train_label = []
#         self.if_val = if_val
#         self.if_train_anomaly = if_train_anomaly
#         if self.dataset == 'mvtec':
#             self.subset = config.SUBSET
#             cls_root = os.path.join(self.img_root, self.subset)
#             if os.path.isdir(cls_root):
#                 cls_img_root = cls_root + '/train/good/'
#                 self.train_path += [os.path.join(cls_img_root, i) for i in os.listdir(cls_img_root)]
#
#         # for cls in os.listdir(self.img_root):
#         #     cls_root = os.path.join(self.img_root, cls)
#         #     if os.path.isdir(cls_root):
#         #         cls_img_root = cls_root + '/train/good/'
#         #         self.train_path += [os.path.join(cls_img_root, i) for i in os.listdir(cls_img_root)]
#
#     def __getitem__(self, index):
#         """
#         :param index:
#         :return: mask, img
#         """
#         name = self.train_path[index]
#         img = Image.open(name).convert('RGB')
#         if self.transform is not None:
#             img = self.transform(img)
#         if not self.if_val:
#             mask = random.choice(self.mask)
#         else:
#             mask = self.mask
#         return mask, img
#
#     def __len__(self):
#         return len(self.train_path)


# aug = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
#
# a = Dataset('/home/yzy/MVTec', aug, 512)
# print(len(a))
# for i, (mask, img) in enumerate(a):
#     if i >= 5:
#         break
#     masked_img = mask * img
#     img = transforms.functional.to_pil_image(img)
#     mask = transforms.functional.to_pil_image(mask)
#     masked_img = transforms.functional.to_pil_image(masked_img)
#     img.save("img"+str(i)+".jpg")
#     mask.save("mask" + str(i) + ".jpg")
#     masked_img.save("masked_img" + str(i) + ".jpg")


# aug = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
#
# a = Testset('/home/yzy/MVTec', aug, 512)
# print(len(a))
# for i, (masks, test, gt, label) in enumerate(a):
#     if i >= 5:
#         break
#     test = transforms.functional.to_pil_image(test)
#     gt = transforms.functional.to_pil_image(gt)
#     test.save("test"+str(i)+".jpg")
#     gt.save("gt" + str(i) + ".jpg")
#     print(label)