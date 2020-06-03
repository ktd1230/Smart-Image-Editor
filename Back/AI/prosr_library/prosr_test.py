from pprint import pprint
from prosr import Phase
from prosr.data import DataLoader, Dataset, DataChunks
from prosr.logger import info
from prosr.metrics import eval_psnr_and_ssim
from prosr.utils import (get_filenames, IMG_EXTENSIONS, print_evaluation, tensor2im)
import numpy as np
import os
import time
import os.path as osp
import prosr
import cv2
import skimage.io as io
import torch
from settings import MODEL_ROOT,OUTPUT_ROOT 
# img_name은 확장자까지 필요..jpg, png 등

def change(img_name,img_path):
    model_path = os.path.join(MODEL_ROOT, "proSR_x2.pth")
    input_path = [img_path +'/'+ img_name]
    target_path = []
    scale = [2, 4, 8]
    scale_idx = 0
    downscale = False
    output_dir = OUTPUT_ROOT
    max_dimension = 0
    padding = 0
    useCPU = False
    # cuda
    checkpoint = torch.load(model_path)
    cls_model = getattr(prosr.models, checkpoint['class_name'])
    model = cls_model(**checkpoint['params']['G'])
    model.load_state_dict(checkpoint['state_dict'])

    # model.load_state_ 모델 데이터 로드하기

    info('phase: {}'.format(Phase.TEST))
    info('checkpoint: {}'.format(osp.basename(model_path)))
    params = checkpoint['params']
    pprint(params)

    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 여기서부터 이제 받아온 데이터로 셋팅
    dataset = Dataset(
        Phase.TEST,
        input_path,
        target_path,
        scale[scale_idx],
        input_size=None,
        mean=params['train']['dataset']['mean'],
        stddev=params['train']['dataset']['stddev'],
        downscale=downscale)

    data_loader = DataLoader(dataset, batch_size=1)

    mean = params['train']['dataset']['mean']
    stddev = params['train']['dataset']['stddev']

    if not osp.isdir(output_dir):
        os.makedirs(output_dir)
    info('Saving images in: {}'.format(output_dir))

    with torch.no_grad():
        if len(target_path):
            psnr_mean = 0
            ssim_mean = 0

        for iid, data in enumerate(data_loader):
            tic = time.time()
            # split image in chuncks of max-dimension
            if max_dimension:
                data_chunks = DataChunks({'input': data['input']}, max_dimension, padding, scale[scale_idx])

                for chunk in data_chunks.iter():
                    input = chunk['input']
                    if not useCPU:
                        input = input.cuda()
                    output = model(input, scale[scale_idx])
                    data_chunks.gather(output)
                output = data_chunks.concatenate() + data['bicubic']
            else:
                input = data['input']
                print("input: ", data['input'])
                if not useCPU:
                    input = input.cuda()
                output = model(input, scale[scale_idx]).cpu() + data['bicubic']
            sr_img = tensor2im(output, mean, stddev)
            toc = time.time()
            if 'target' in data:
                hr_img = tensor2im(data['target'], mean, stddev)
                psnr_val, ssim_val = eval_psnr_and_ssim(
                    sr_img, hr_img, scale[scale_idx])
                print_evaluation(
                    osp.basename(data['input_fn'][0]), psnr_val, ssim_val,
                    iid + 1, len(dataset), toc - tic)
                psnr_mean += psnr_val
                ssim_mean += ssim_val
            else:
                print_evaluation(
                    osp.basename(data['input_fn'][0]), np.nan, np.nan, iid + 1,
                    len(dataset), toc - tic)

            # 출력
            fn = osp.join(output_dir, 'result_'+osp.basename(data['input_fn'][0]))
            io.imsave(fn, sr_img)
            # ir = io.imread(fn)
            # w , h, s = ir.shape
            # nw=int(w/2)
            # nh=int(h/2)
            # resize_img = cv2.resize(ir, (0, 0),fx=0.5,fy=0.5, interpolation=cv2.INTER_AREA)
            # io.imsave(fn,resize_img)


        if len(target_path):
            psnr_mean /= len(dataset)
            ssim_mean /= len(dataset)
            print_evaluation("average", psnr_mean, ssim_mean)
        return 'result_'+osp.basename(data['input_fn'][0])
# if __name__ == '__main__':
#     change('waterfall.jpg')