import torch
from PIL import Image
from torchvision import transforms,models
import matplotlib.pyplot as plt
from config import get_maskrcnn_cfg
import random
import colorsys
import numpy as np
from utils import date2str

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    print(mask)
    for c in range(3):
        image[c,:, :] = np.where(mask == 0,
                                  image[c,:, :] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  0)

    return image

def image_load(image,image_saved_path):
    # Read image using PIL
    im = Image.open(image_saved_path + image)
    return im

def transform_to_tensor(im):
    loader = transforms.Compose([
        # transforms.Resize(imsize),  # 입력 영상 크기를 맞춤
        transforms.ToTensor()])
    tensor = loader(im).unsqueeze(0)
    return tensor

def load_model(AI_directory_path=None):
    if AI_directory_path is None:
        return models.detection.maskrcnn_resnet50_fpn(pretrained=True)

def img_frombytes(data):
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)


def save_segments(original_image, segments, image_saved_path):
    original_image = original_image.split(".")[0]
    output_names = []
    time = date2str()
    for k, segment in enumerate(segments):
        pil_object = Image.fromarray(segment.astype('uint8'),'RGB')
        output_name = original_image +"segment{0}_{1}.jpg".format(k, time)
        output_names.append(output_name)
        pil_object.save(image_saved_path + output_name)
    return output_names

def save_masks(original_image,masks,image_saved_path):
    original_image = original_image.split(".")[0]
    output_names = []
    time = date2str()
    for k,mask in enumerate(masks):
        def img_frombytes(data):
            size = data.shape[::-1]
            databytes = np.packbits(data, axis=1)
            return Image.frombytes(mode='1', size=size, data=databytes)
        pil_object = img_frombytes(mask)
        output_name = original_image + "mask{0}_{1}.jpg".format(k, time)
        output_names.append(output_name)
        pil_object.save(image_saved_path + output_name)
    return output_names

def gather_mask_beyond_threshold(output,threshold=0.5):

    scores = list(output['scores'].detach().cpu().numpy())
    pred_t = [scores.index(x) for x in scores if x > threshold][-1]
    masks = (output['masks'] < 0.5).squeeze().detach().cpu().numpy().astype(np.uint8)*255
    masks = masks[:pred_t + 1]
    return masks

def get_segment_from_mask(masks,original_image):
    segmented_images = []
    numpy_image = np.array(original_image).astype(np.uint8).transpose((2, 0, 1))
    N = masks.shape[0]
    if N :
        colors = random_colors(N)
        for k,mask in enumerate(masks):
            color = colors[k]
            new_image = numpy_image.copy()
            segment = apply_mask(new_image,mask,color).transpose(1,2,0)
            segmented_images.append(segment)
    return segmented_images

def predict(image_name,image_saved_path,AI_directory_path="temp"):

    # param :
    # image_name : 백에서 넘겨준 file의 이름
    # image_saved_path : 이미지를 저장하는 백의 경로

    #read image and transform
    original_image = image_load(image_name,image_saved_path)

    image = transform_to_tensor(original_image)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #load model
    model = load_model()

    # model run environment modify
    model.to(device)
    image = image.to(device)


    # model 수행
    with torch.no_grad():
        model.eval()
        output = model(image)[0]

    # output에서 mask 결과만 모으기
    # output에는 bounding box output, mask output, 등등
    reliable_masks = gather_mask_beyond_threshold(output, threshold=0.5)

    #display_instances(image=original_image,masks=reliable_masks, title="asd",figsize=(16, 16), ax=None)
    # prediction 결과의 정확도가 threshold 이상인 마스크들만 모으기

    # 얻어낸 마스크와 원본이미지를 곱해서 특정 객체의 segment 따기
    target_segments = get_segment_from_mask(reliable_masks,original_image)

    # target_segents 을 image_path에 저장하고 백으로 넘겨줄 이름들을 리턴받아옴
    saved_segment_names = save_segments(image_name,target_segments,image_saved_path)

    # 태동이가 하는 inpainting이 mask를 요구하기 때문에 mask도 저장
    saved_mask_names = save_masks(image_name,reliable_masks,image_saved_path)

    return saved_segment_names,saved_mask_names



def main():
    import os
    predict("chichi.jpg", os.getcwd() + "\dataload\education\chichi")
    return


if __name__ == "__main__":
    #main()
    image_root = "C:\\Users\\multicampus\\Downloads\\asdf\\"
    #image = "000000000009.jpg"
    image = "000000000036.jpg"
    predict(image,image_root)

