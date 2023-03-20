import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--network', default='VGG19', type=str, help='TResNet or VGG19')
parser.add_argument('--multigpu', default=False, type=str, help='Use multi-gpu to train model or not')
parser.add_argument('--pretrained', default=True, type=str, help='Use pretrained model or not')
parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
parser.add_argument('--workers', default=12, type=int, help='number of data loading workers')
parser.add_argument('--dataset', default='NUSWIDE', type=str, help='NUSWIDE or MSCOCO')
parser.add_argument('--batch_size', type=int, default=96, help='input batch size')
parser.add_argument('--start_epoch', type=int, default=0, help='Resume training at this iter')
parser.add_argument('--nepoch', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--test_epoch', type=int, default=5, help='number of epochs to test')
parser.add_argument('--nhead', type=int, default=16, help='number of heads')
parser.add_argument('--ndepth', type=int, default=3, help='number of depth')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate to train')
#
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--nclass_all', type=int, default=1006, help='number of all classes')
parser.add_argument('--nseen_class', type=int, default=925, help='number of seen classes')
parser.add_argument('--nfeat_input', type=int, default=1792, help='number of feature input')
parser.add_argument('--nfeat_output', type=int, default=2048, help='number of feature output')

#
parser.add_argument('--input_size', type=int, default=224, help='size of the input images')
parser.add_argument('--num_rows', type=int, default=7, help='number of output rows')
parser.add_argument('--wordvec_dim', type=int, default=300, help='dimension of word vectors')
parser.add_argument('--weight', type=float, default=0.4, help='loss weights')
#
parser.add_argument('--log_name', default='Channel_MLZSL', help='name for saving logging')
parser.add_argument('--save_folder', default='weights/', help='Directory for saving checkpoint models')

parser.add_argument('--manualSeed', type=int, help='manual seed')
#nus-wide
parser.add_argument('--train_json', type=str, default='custom_data_train', help='training json filename')
parser.add_argument('--test_json', type=str, default='custom_data_test', help='testing json filename')
parser.add_argument('--dataset_path', type=str, default='data/NUS_WIDE', help='dataset path')
parser.add_argument('--path_output', type=str, default='./output', help='output path')
parser.add_argument('--image_dir', type=str, default='data/NUS_WIDE/Flickr', help='foldername containg all the images')
parser.add_argument('--output_dir', type=str, default='data/NUS_WIDE/custom_data_jsons', help='foldername containg generated jsons')

#
parser.add_argument('--trained_model',
                    default='weights/ChannelZSL_MLPGA224_NUSWIDE_7_0.4_96_final.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_path',
                    default='results/', type=str,
                    help='Save results')


opt = parser.parse_args()
