import argparse
import template
from pathlib import Path

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='../dataset',
                    help='dataset directory')
parser.add_argument('--dir_demo',
                    type=str,
                    default=(Path(__file__).parent.absolute() / "..\\..\\Django\\media").resolve().__str__(),
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/801-810',
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', type=str, default='4',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# Model specifications
parser.add_argument('--model', default='EDSR',
                    help='model name')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')

# Option for Residual channel attention network (RCAN)
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
# parser.add_argument('--epochs', type=int, default=2,
#                     help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')

# args = parser.parse_args()
args, _ = parser.parse_known_args()
template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False


def set_setting_value_edsr(images, root_path, ai_directory_path, use_cpu=False):
    """
    :param images: 이미지 이름을 의미하며 복수개의 이미지 이름(str을 원소로 갖는 list)
    :param root_path: 이미지가 저장된 디렉토리(str)
    :param ai_directory_path: EDSR model의 경로. model 이름까지 함께 있어야 한다. (*.pt)
    :param use_cpu: EDSR을 수행할 때 CPU 사용 여부. (False 값일 때 GPU를 사용한다.)
    :return: 별도의 return은 존재하지 않음
    """
    args.G0 = 64
    args.RDNconfig = 'B'
    args.RDNkSize = 3
    args.act = 'relu'
    args.batch_size = 16
    args.betas = (0.9, 0.999)
    args.chop = False
    args.cpu = use_cpu
    args.data_range = '1-800/801-810'
    args.data_test = ['Demo']
    args.data_train = ['DIV2K']
    args.debug = False
    args.decay = '200'
    args.dilation = False
    args.dir_data = 0
    args.dir_demo = './test'
    args.epochs = 300
    args.epsilon = 1e-08
    args.ext = 'sep',
    args.extend = '.'
    args.gamma = 0.5
    args.gan_k = 1
    args.gclip = 0
    args.load = ''
    args.loss = '1*L1'
    args.lr = 0.0001
    args.model = 'EDSR'
    args.momentum = 0.9
    args.n_GPUs = 1
    args.n_colors = 3
    args.n_feats = 64
    args.n_resblocks = 16
    args.n_resgroups = 10
    args.n_threads = 6
    args.no_augment = False
    args.optimizer = 'ADAM'
    args.patch_size = 192
    # args.pre_train = '../experiment/edsr_baseline_x2/model/model_best.pt'  # model path
    # args.pre_train = "C:\\s02p31c101\\Back\\AI\\experiment\\edsr_baseline_x2\\model\\model_best.pt"
    args.pre_train = ai_directory_path
    args.precision = 'single'
    args.print_every = 100
    args.reduction = 16
    args.res_scale = 1
    args.reset = False
    args.resume = 0
    args.rgb_range = 255
    args.save = '.'
    args.save_gt = False
    args.save_models = False
    args.save_results = True
    args.scale = [2]
    args.seed = 1
    args.self_ensemble = False
    args.shift_mean = True
    args.skip_threshold = 100000000.0
    args.split_batch = 1
    args.template = '.'
    args.test_every = 1000
    args.test_only = True
    args.weight_decay = 0

    # 정적으로 할당된 변수를 이용하지 않을 때는 아래의 코드를 이용한다.
    args.image_name_list = [images]  # ex) print(args.image_name_list) : ["0894x2.png", "0920x2.png"]
    args.dir_demo = root_path  # ex) print(root_path) : "./test"

    print("args.image_name_list:{}".format(args.image_name_list))
    print("args.dir_demo:{}".format(args.dir_demo))

    # 테스트를 위해 아래의 코드를 이용할 수 있다.
    # args.image_name_list = ["0894x2.png", "0920x2.png"]
    # args.dir_demo = "C:\\s02p31c101\\Back\\AI\\edsr_library\\test"
