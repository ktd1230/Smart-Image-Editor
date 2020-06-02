import torch
# import config

import sys
import torch

import utility
import data
import model
import loss
from option import args
from option import set_setting_value_edsr
from trainer import Trainer
import timeit

from PIL import Image
import numpy as np
from math import floor


def png_alpha_channel_remove(image_name, root_path):
    """
    아래의 모든 매개변수는 predict 함수의 매개변수에 종속된다.
    :param image_name: image의 이름
    :param root_path: image가 저장된 디렉토리
    :return: 별도의 반환 데이터는 존재하지 않으나 open된 파일이 32-bit PNG파일이라면 24-bit PNG파일로 변환하여 저장한다.
    """
    image_full_path = root_path + "\\"
    # print("image_full_path: {}".format(image_full_path))
    img = Image.open(image_full_path + image_name)
    image_array = np.array(img)
    # print(image_array.shape)
    if image_array.shape[2] > 3:  # Alpha 값이 존재하는 PNG type의 이미지인 경우, Alpha 채널을 제거해야 함
        # print("Try alpha channel remove ...")
        image_array = image_array[..., :3]
        img = Image.fromarray(image_array)
        # print("done")
        img.save(image_full_path + image_name)
        img.close()


def downscale_by_ratio(image_name, root_path, ratio, method=Image.BICUBIC):
    """
    :param image_name: image의 이름
    :param root_path: image가 저장된 디렉토리
    :return: 별도의 반환 데이터는 존재하지 않으나 ratio 비율로 축소한 이미지를 저장한다.
    """
    if ratio == 1:
        return
    image_full_path = root_path + "\\"
    img = Image.open(image_full_path + image_name)
    width, height = img.size
    width, height = floor(width / ratio), floor(height / ratio)
    # print("width:{}, height:{}".format(width, height))
    img.resize((width, height), method).save(image_full_path + image_name)
    img.close()


def predict(images="", root_path="", ai_directory_path="", model_type="EDSR"):
    """
    :param images: image의 이름 (특화 프로젝트 때, 복수의 이미지 파일을 받아서 images로 명명됨)
    :param root_path: image가 저장된 디렉토리
    :param AI_directory_path: 모델이 저장된 디렉토리
    :param model_type:
    :return: 생성된 이미지 파일 경로+이름 (list)
    """
    if model_type == "EDSR":
        png_alpha_channel_remove(images, root_path)
        set_setting_value_edsr(images, root_path, ai_directory_path, use_cpu=False)
        torch.manual_seed(args.seed)
        checkpoint = utility.checkpoint(args)
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            result = t.test()  # return value is saved image path(and image name). list type.
            checkpoint.done()

            # for file_name in result 형태의 for문 형태는 result의 str원소를 변경할 수 없다.
            for i in range(len(result)):
                result[i] = result[i][result[i].rfind("\\") + 1:]
            return result  # `media` 디렉토리 내부에 존재하는 결과물 파일 이름을 반환

            # # result 값이 변경되지 말아야 하는 경우 아래의 코드를 대신 사용한다.
            # only_file_name_list = []  # 새로운 반환 리스트 생성
            # for file_name in result:
            #     # file_name = file_name[file_name.rfind("\\") + 1:]
            #     # 위 코드의 경우 참조 형태가 아니므로 file_name의 변경이 result 원소에 영향을 주지 않는다.
            #     only_file_name_list.append(file_name[file_name.rfind("\\") + 1:])
            # return only_file_name_list  # `media` 디렉토리 내부에 존재하는 결과물 파일 이름을 반환


def main():
    predict()


if __name__ == "__main__":
    main()
