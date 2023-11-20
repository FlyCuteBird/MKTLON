import argparse
# Hyper Parameters setting
def parse_opt():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='./data/',
                        help='path to datasets')
    parser.add_argument('--data_name', default='f30k_precomp',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=768, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=10, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=100, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=3000000, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='./runs/runX/log',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--result_name', default='./runs/runX/result',
                        help='Path to save matching result.')
    parser.add_argument('--model_name', default='./runs/runX/checkpoint',
                        help='Path to save the model.')
    parser.add_argument('--resume', default='./runs/runX/checkpoint/Position_relation/checkpoint_000.pth.tar', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--bi_gru', action='store_false',
                        help='Use bidirectional GRU.')
    parser.add_argument('--feat_dim', default=32, type=int,
                        help='Dimensionality of the similarity embedding.')
    parser.add_argument('--num_block', default=32, type=int,
                        help='Dimensionality of the similarity embedding.')
    parser.add_argument('--hid_dim', default=32, type=int,
                        help='Dimensionality of the hidden state during graph convolution.')
    parser.add_argument('--out_dim', default=1, type=int,
                        help='Dimensionality of the hidden state during graph convolution.')
    parser.add_argument('--Rank_Loss', default='DynamicTopK_Negative',
                        help='DynamicTopK_Negative||Hardest_Negative||Hard_Negative')
    parser.add_argument('--region_relation', default='NTN_relation',
                        help='NTN_relation||Position_relation.')
    parser.add_argument('--Pieces', default=4, type=int,
                        help=' the pieces of Meanout.')
    parser.add_argument('--Matching_direction', default='t2i',
                        help='i2t||t2i||ti, image-to-text matching or text-to-image matching')

    parser.add_argument('--N', default=3, type=int,
                        help=' The number of encoding layers.')
    parser.add_argument('--belt_alpha', default=0.3, type=float,
                        help=' the result of the belt/alpha.')
    parser.add_argument('--thres', default=0.2, type=float,
                        help=' The number of encoding layers.')
    parser.add_argument('--thres_safe', default=0.2, type=float,
                        help=' The number of encoding layers.')
    parser.add_argument('--mean_neg', default=0, type=float,
                        help=' The number of encoding layers.')
    parser.add_argument('--mean_pos', default=0, type=float,
                        help=' The number of encoding layers.')
    parser.add_argument('--stnd_neg', default=0, type=int,
                        help=' The number of encoding layers.')
    parser.add_argument('--stnd_pos', default=0, type=int,
                        help=' The number of encoding layers.')

    parser.add_argument('--count_num', action='store_false',
                        help=' The number of encoding layers.')

    parser.add_argument('--positive_example', default=[], type=list,
                        help=' The number of encoding layers.')
    parser.add_argument('--negative_example', default=[], type=list,
                        help=' The number of encoding layers.')


    opt = parser.parse_args()

    return opt
