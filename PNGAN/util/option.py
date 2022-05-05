import argparse
from util import template

parser = argparse.ArgumentParser(description='RIDNET')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=8, help='number of threads for dataset loading')
parser.add_argument('--cpu', action='store_true', default=True, help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='./Datasets', help='dataset directory')
parser.add_argument('--data_train', type=str, default='SIDD',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='SIDD',
                    help='test dataset name')
parser.add_argument('--benchmark_noise', action='store_true',
                    help='use noisy benchmark sets')
parser.add_argument('--partial_data', action='store_true', help='Only use part data')
parser.add_argument('--n_train', type=int, default=20000,
                    help='number of training set')
parser.add_argument('--n_val', type=int, default=2000,
                    help='number of validation set')
parser.add_argument('--ext', type=str, default='sep_reset',
                    help='dataset file extension')
parser.add_argument('--noise_g', default='1',
                    help='Gaussian Noise')

# parser.add_argument('--scale', default='4', help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=128,
                    help='output patch size')
parser.add_argument('--predict_patch_size', type=int, default=800,
                    help='predict patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--noise', type=int, default=50, help='Gaussian noise std.')
parser.add_argument('--chop', action='store_true', help='enable memory-efficient forward')

# Model specifications
parser.add_argument('--model', default='RIDNET', help='model name')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='./experiment/ridnet.pt',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true', default=True, help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--lr', type=float, default=2e-4,
                    help='initial learning rate')
parser.add_argument('--lr_min', type=float, default=1e-7,
                    help='min learning rate')
parser.add_argument('--lr_decay_step', type=int, default=1e5,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.9999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0.8,
                    help='weight decay')

# options feature channel reduction
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='./', help='file name to save')
parser.add_argument('--load_models', action='store_true', help='Load Pretrained Model')
parser.add_argument('--load_best', action='store_true', help='Load the best model you trained')
parser.add_argument('--load_dir', type=str, default='.', help='Place you put your model')
parser.add_argument('--timestamp', type=int, default=None, help='The timestamp for your model')
parser.add_argument('--load_epoch', type=int, default=None, help='The Epoch You want to load')

parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--print_model', action='store_true',
                    help='print model')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true', help='save output results')

# options for test
parser.add_argument('--testpath', type=str, default='./test',
                    help='dataset directory for testing')
parser.add_argument('--savepath', type=str, default='./save',
                    help='places for save output image')

args = parser.parse_args(args=[])
template.set_template(args)

args.noise_g = list(map(lambda x: int(x), args.noise_g.split('+')))

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
