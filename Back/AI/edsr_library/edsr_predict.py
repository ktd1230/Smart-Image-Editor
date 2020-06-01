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


def predict(images="", root_path="", ai_directory_path="", model_type="EDSR"):
    """
    :param images: image의 이름 (특화 프로젝트 때, 복수의 이미지 파일을 받아서 images로 명명됨)
    :param root_path: image가 저장된 디렉토리
    :param AI_directory_path: 모델이 저장된 디렉토리
    :param model_type:
    :return: 생성된 이미지 파일 경로+이름 (list)
    """

    if model_type == "EDSR":
        set_setting_value_edsr(images, root_path, ai_directory_path)
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
