


def predict(images="", root_path="", AI_directory_path=""):
    """
    :param images: image의 이름 (특화 프로젝트 때, 복수의 이미지 파일을 받아서 images로 명명됨)
    :param root_path: image가 저장된 디렉토리
    :param AI_directory_path: 모델이 저장된 디렉토리
    :param model_type:
    :return: 생성된 이미지 파일 경로+이름 (list)
    """

    if model_type == "EDSR":
        set_setting_value_edsr(images, root_path)
        torch.manual_seed(args.seed)
        checkpoint = utility.checkpoint(args)
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            result = t.test()  # return value is saved image path(and image name). list type.
            checkpoint.done()
            return result  # list

def main():
    predict()


if __name__ == "__main__":
    main()
