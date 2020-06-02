import argparse
import torch
from torchvision import transforms

# import opt
# from evaluation import evaluate
# from util.io import load_ckpt
import math
import torch.nn as nn
import torch.nn.functional as F
import random
from PIL import Image
from glob import glob
from torchvision.utils import make_grid
from torchvision.utils import save_image

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun

class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask

class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',
                 conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask

class PConvUNet(nn.Module):
    def __init__(self, layer_size=7, input_channels=3, upsampling_mode='nearest'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7')
        self.enc_2 = PCBActiv(64, 128, sample='down-5')
        self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_4 = PCBActiv(256, 512, sample='down-3')
        for i in range(4, self.layer_size):
            name = 'enc_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512, 512, sample='down-3'))

        for i in range(4, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512 + 512, 512, activ='leaky'))
        self.dec_4 = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1 = PCBActiv(64 + input_channels, input_channels,
                              bn=False, activ=None, conv_bias=True)

    def forward(self, input, input_mask):
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N

        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev])
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        # concat upsampled output of h_enc_N-1 and dec_N+1, then do dec_N
        # (exception)
        #                            input         dec_2            dec_1
        #                            h_enc_7       h_enc_8          dec_8

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)

            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            h_mask = F.interpolate(
                h_mask, scale_factor=2, mode='nearest')

            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)

        return h, h_mask

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()

class Places2(torch.utils.data.Dataset):
    def __init__(self, img_root, img_names, mask_names, img_transform, mask_transform):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.paths = []
        self.mask_paths = []

        # use about 8M images in the challenge dataset
        for i, image in enumerate(img_names):
            print(i)
            print(image)
            print(self.paths)
            self.paths.append('{:s}/{:s}'.format(img_root, image))
        print(self.paths)
        print(img_root)
        for i, mask in enumerate(mask_names):
            self.mask_paths.append('{:s}/{:s}'.format(img_root, mask))
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

def load_ckpt(ckpt_name, models, optimizers=None):
    print(ckpt_name)
    ckpt_dict = torch.load(ckpt_name)
    for prefix, model in models:
        assert isinstance(model, nn.Module)
        model.load_state_dict(ckpt_dict[prefix], strict=False)
    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])
    return ckpt_dict['n_iter']

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor(STD) + torch.Tensor(MEAN)
    x = x.transpose(1, 3)
    return x
import os
BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

def predict(images, masks, root_path, AI_directory_path, model_type="life"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_root', type=str, default='./mask')
    print(masks)
    print(BASE_DIR)

    # args = parser.parse_args()

    device = torch.device('cuda')

    size = (256, 256)
    img_transform = transforms.Compose(
        [transforms.Resize(size=size), transforms.ToTensor(),
         transforms.Normalize(mean=MEAN, std=STD)])
    mask_transform = transforms.Compose(
        [transforms.Resize(size=size), transforms.ToTensor()])

    dataset_val = Places2(root_path, images, masks, img_transform, mask_transform)
    model = PConvUNet().to(device)
    load_ckpt(AI_directory_path, [('model', model)])

    model.eval()

    image, mask, gt, ori_size = zip(*[dataset_val[i] for i in range(1)])
    image = torch.stack(image)
    mask = torch.stack(mask)
    ori_size = ori_size[0]
    gt = torch.stack(gt)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output

    grid = make_grid(
        torch.cat((unnormalize(image), mask, unnormalize(output),
                   unnormalize(output_comp), unnormalize(gt)), dim=0))
    save_image(unnormalize(output), BASE_DIR + '\\Django\\media\\result.jpg')
    print(ori_size)
    img_transform = transforms.Compose([transforms.Resize((ori_size[1], ori_size[0])), transforms.ToTensor()])
    output = Image.open(BASE_DIR + '\\Django\\media\\result.jpg')
    print(output)
    output = img_transform(output)
    save_image(output, BASE_DIR + '\\Django\\media\\result.jpg')

    return 'result.jpg'


# root_path: image가 저장된 폴더
# images: image의이름(특화프로젝트때, 복수의이미지파일을받아서images였음)
# AI_directory_path: 저장된모델의디렉토리
# from config import get_config
#
# def predict(images, root_path, AI_directory_path, model_type="life"):
#     # 0. config -> model 설정 관련 변수를 관리하는 객체 [config.py] 호출
#     config = get_config()
#     # 1. 모델과 관련된 변수들 정리
#     vocab = load_voca(AI_directory_path + config['caption_vocab_path'])
#     caption_embed_size = config['caption_embed_size']
#     caption_hidden_layer = config['caption_hidden_layer']
#     caption_hidden_size = config['caption_hidden_size']
#     caption_encoder_path = AI_directory_path + config['caption_encoder_path']
#     caption_decoder_path = AI_directory_path + config['caption_decoder_path']
#     max_sequence_len = 30  # default value
#
#     # 2. hardware 자원 설정
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     # 3. 이미지를 로드하고 pytorch의 Tensor로 바꾸기 위한 transform 작성
#     transform = torch_transform.Compose([
#         torch_transform.ToTensor(),
#         torch_transform.Normalize(mean=(0.4444, 0.4215, 0.3833), std=(0.2738, 0.2664, 0.2766))])
#     images = load_image(images, root_path, transform)
#
#     # 4. 모델 선언
#     encoder = EncoderCNN(caption_embed_size)
#     decoder = DecoderRNN(caption_embed_size, len(vocab), caption_hidden_layer, caption_hidden_size)
#
#     # 5. 모델 로드
#     encoder.load_state_dict(torch.load(caption_encoder_path, map_location=device))
#     decoder.load_state_dict(torch.load(caption_decoder_path, map_location=device))
#
#     # 6. 모델을 테스트 모드로 설정
#     encoder.eval()
#     decoder.eval()
#
#     # 7. 모델을 target 디바이스로 변환
#     encoder.to(device)
#     decoder.to(device)
#
#     # 8. 이미지를 target 디바이스로 copy
#     images = images.to(device)
#
#     # 9. 모델실행
#     features = encoder(images)
#     states = None
#     predicted_index = []
#     lstm_inputs = features.unsqueeze(1)
#
#     for i in range(max_sequence_len):
#         outputs, states = decoder.lstm(lstm_inputs, states)
#         # outputs을 linear 레이어의 인풋을 위해 2차원 배열로 만들어 줘야함
#         outputs = outputs.squeeze(1)
#         scores_per_batch = decoder.score_layer(outputs)
#         values, predicted = scores_per_batch.max(1)
#         predicted_index.append(predicted)
#         lstm_inputs = decoder.embed(predicted)
#         lstm_inputs = lstm_inputs.unsqueeze(1)
#
#     predicted_index = torch.stack(predicted_index, dim=1)
#     predicted_index = predicted_index.cpu().numpy()
#
#     result_captions = []
#     for wordindices in predicted_index:
#         text = ""
#         for index in wordindices:
#             word = vocab.idx2word[index]
#             if word == '<end>':
#                 break
#             if word == '<unk>' or word == '<start>':
#                 continue
#             text += word + " "
#         result_captions.append(text)
#
#     print("result_caption : ", result_captions)
#     # 1. translate captions to korean
#
#     korean_sentences = []
#     for sent in result_captions:
#         translate_result = get_translate(sent)
#         if translate_result != -1:
#             translate_result = re.sub(r'\.', '', translate_result)
#             korean_sentences.append(translate_result)
#     print("result_korean : ", korean_sentences)
#
#     kogpt2_config = get_kog_config()
#     if model_type == "life":
#         kogpt2_model_path = AI_directory_path + config['kogpt_life_model_path']
#     elif model_type == "story":
#         kogpt2_model_path = AI_directory_path + config['kogpt_story_model_path']
#     else:
#         kogpt2_model_path = AI_directory_path + config['kogpt_model_path']
#     kogpt2_vocab_path = AI_directory_path + config['kogpt_vocab_path']
#     kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
#     kogpt2model.load_state_dict(torch.load(kogpt2_model_path, map_location=device))
#
#     kogpt2model.to(device)
#     kogpt2model.eval()
#     vocab = nlp.vocab.BERTVocab.from_sentencepiece(kogpt2_vocab_path,
#                                                    mask_token=None,
#                                                    sep_token=None,
#                                                    cls_token=None,
#                                                    unknown_token='<unk>',
#                                                    padding_token='<pad>',
#                                                    bos_token='<s>',
#                                                    eos_token='</s>')
#     tok = SentencepieceTokenizer(kogpt2_vocab_path)
#
#     korean_preprocess(korean_sentences)
#     gpt_result = naive_prediction(korean_sentences, tok, vocab, device, kogpt2model, model_type)
#     korean_postprocess(gpt_result)
#     result = []
#     make_sentence(gpt_result, "", result, 0)
#     result.sort(key=lambda item: (-len(item), item))
#     result_len = len(result)
#     if result_len > 11:
#         result_len = 11
#     result = result[1:result_len]
#
#     # 10. 모델의 수행 결과를 리턴
#     return result


