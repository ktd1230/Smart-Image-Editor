import torch
from PIL import Image
from torchvision import transforms,models
from config import get_maskrcnn_cfg
def image_load(image,image_saved_path):
    # Read image using PIL
    im = Image.open(image_saved_path + "\\" + image)

    # transform image to tensor using torchvision.transforms
    # imsize = (256, 256)
    loader = transforms.Compose([
        # transforms.Resize(imsize),  # 입력 영상 크기를 맞춤
        transforms.ToTensor()])
    tensor = loader(im).unsqueeze(0)
    # print(tensor)
    # print(tensor.shape)
    return tensor

def load_model(AI_directory_path=None):
    if AI_directory_path is None:
        return models.detection.maskrcnn_resnet50_fpn(pretrained=True)


def save_output(output_name,outputs,image_saved_path):
    # param
    # outputs : tensor list
    # output_name : will be saved
    output_names = []
    pass

    # Using output_name ,save tensor outputs as image like 'output_name_1.jpg' at image_saved_path while transforming tensor to jpg

    # return output_names

def gather_mask_beyond_threshold(outputs,scores,threshold=0.5):
    reliable_outputs = []
    # 신뢰도가 threshold 이상인 객체만 모으기
    for i in range(0, scores.size(0)):
        if scores[i] > threshold:
           reliable_outputs.append(outputs[i])
    print(reliable_outputs)


    return reliable_outputs

def get_segment_from_mask(masks,original_image):
    pass

def predict(image_name,image_saved_path,AI_directory_path="temp"):

    # param :
    # image_name : 백에서 넘겨준 file의 이름
    # image_saved_path : 이미지를 저장하는 백의 경로

    #read image and transform
    image = image_load(image_name,image_saved_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #load model
    model = load_model()

    # model run environment modify
    model.to(device)
    image = image.to(device)


    # model 수행
    with torch.no_grad():
        model.eval()
        output = model(image)

    masks = None
    # output에서 mask 결과만 모으기
    # output에는 bounding box output, mask output, 등등
    masks = output[0]["masks"]
    scores =  output[0]["scores"]

    # prediction 결과의 정확도가 threshold 이상인 마스크들만 모으기
    reliable_masks = gather_mask_beyond_threshold(masks,scores,threshold=0.5)
    return

    # 얻어낸 마스크와 원본이미지를 곱해서 특정 객체의 segment 따기
    target_segments = get_segment_from_mask(masks,image)

    # target_segents 을 image_path에 저장하고 백으로 넘겨줄 이름들을 리턴받아옴

    output_name = " " #time을 이용해서 output_name 정하기
    saved_segment_names = save_output(output_name,target_segments,image_saved_path)

    # 태동이가 하는 inpainting이 mask를 요구하기 때문에 mask도 저장
    output_name = " " #time을 이용해서 output_name 정하기
    saved_mask_names = save_output(output_name,reliable_masks,image_saved_path)

    return saved_segment_names,saved_mask_names



def main():
    import os
    predict("chichi.jpg", os.getcwd() + "\dataload\education\chichi")
    return


if __name__ == "__main__":
    main()
