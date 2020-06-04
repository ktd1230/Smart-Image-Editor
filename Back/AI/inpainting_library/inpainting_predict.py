import torch
from torchvision import transforms

import opt
from places2 import Places2
from evaluation import evaluate
from net import PConvUNet
from util.io import load_ckpt
from util.io import get_state_dict_on_cpu

def predict(image, mask, root_path, AI_directory_path, model_type="life"):

    device = torch.device("cpu")

    size = (256, 256)
    img_transform = transforms.Compose(
        [transforms.Resize(size=size), transforms.ToTensor(),
         transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
    mask_transform = transforms.Compose(
        [transforms.Resize(size=size), transforms.ToTensor()])

    dataset_val = Places2(root_path, image, mask, img_transform, mask_transform)
    model = PConvUNet().to(device)
    load_ckpt(device, AI_directory_path, [('model', model)])
    # get_state_dict_on_cpu(model)

    model.eval()

    evaluate(model, dataset_val, device, image.split('.')[0] + 'result.jpg')

    return image.split('.')[0] + 'result.jpg'

