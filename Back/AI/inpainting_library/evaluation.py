import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image

from util.image import unnormalize

import os
BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

def evaluate(model, dataset, device, filename):
    print(BASE_DIR)
    image, mask, gt, ori_size = zip(*[dataset[i] for i in range(1)])

    image = torch.stack(image)
    mask = torch.stack(mask)
    ori_size = ori_size[0]

    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))

    RESULT_DIR = BASE_DIR + '\\Django\\media\\'

    save_image(unnormalize(output), RESULT_DIR + filename)
    print(ori_size)
    img_transform = transforms.Compose([transforms.Resize((ori_size[1], ori_size[0])), transforms.ToTensor()])
    output = Image.open(RESULT_DIR + filename)
    print(output)
    output = img_transform(output)
    save_image(output, RESULT_DIR + filename)
