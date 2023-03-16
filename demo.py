import matplotlib.pyplot as plt
import torch
from PIL import Image

# a = torch.randn(512, 512)
# a = a.numpy()
# plt.plot()
# plt.imshow(a)
# plt.show()

# import cv2
# import numpy as np
#
# img = np.random.randint(255, size=(300, 600, 3))
#
# isWritten = cv2.imwrite('D:/image-2.png', img)

a = torch.load ('/home/yzy/SCADN/checkpoints/dataset_mvtec/subset_carpet/anomaly_num_30/seed_10/lr_0.0001/d2glr_0.1/epoch_1.pth', 'cpu')
a1 = a['model_g']
print(a1.keys())
# b = torch.load ('/home/yzy/SCADN/checkpoints/dataset_mvtec/subset_carpet/anomaly_num_30/seed_10/lr_0.0001/d2glr_0.1/epoch_1.pth','cpu)('model_g']
# print(b)
# assert False
# for k in a1.keys:
#     if k not in b.keys():
#         print(k)
#     else:
#         print(a 1kl == b[k])