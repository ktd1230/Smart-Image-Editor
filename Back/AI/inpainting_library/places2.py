import random
import torch
from PIL import Image

class Places2(torch.utils.data.Dataset):
    def __init__(self, img_root, img_name, mask_name, img_transform, mask_transform):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        self.paths = []
        self.mask_paths = []

        # use about 8M images in the challenge dataset

        self.paths.append('{:s}/{:s}'.format(img_root, img_name))
        print(self.paths)
        print(img_root)

        self.mask_paths.append('{:s}/{:s}'.format(img_root, mask_name))
        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        ori_size = gt_img.size
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask.convert('RGB'))
        return gt_img * mask, mask, gt_img, ori_size

    def __len__(self):
        return len(self.paths)
