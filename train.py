import os
import argparse
from script import helpers
import tensorflow as tf

from script.data_loader import get_generator
from script.model_trainer import ModelTrainer3D, ModelTrainer2D

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument('--exp', type=str, default="exp_s2_edge")
    parser.add_argument('--num_epochs', type=int, default=501)
    # parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--validate_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)

    # data
    parser.add_argument('--patch_x', type=int, default=256)
    parser.add_argument('--patch_y', type=int, default=256)
    parser.add_argument('--patch_z', type=int, default=16)
    parser.add_argument('--data_list_path', type=str, default='./data_uci/data/label') # path for female.txt ...
    parser.add_argument('--data_path', type=str, default="./data_uci/data_uci_converted")
    parser.add_argument('--sex', type=int, default=0, help="0 for all, 1 for male, 2 for female")

    # cpu
    parser.add_argument('--n_workers', type=int, default=6)

    # gpu
    parser.add_argument('--gpu', type=str, default="0")

    # data augmentation
    parser.add_argument('--patch_center_dist_from_border', type=tuple, default=(256//2-5, 256//2-5,16//2))
    parser.add_argument('--do_random_crop', type=helpers.str2bool, default=False)
    parser.add_argument('--do_rotation', type=helpers.str2bool, default=False)
    parser.add_argument('--rotation_angle',type=int,default=10)
    parser.add_argument('--do_elastic_deform', type=helpers.str2bool, default=False)
    parser.add_argument('--elastic_deform', type=tuple, default=(0, 0.05))
    parser.add_argument('--do_scale', type=helpers.str2bool, default=False)
    parser.add_argument('--scale_factor', type=tuple, default=(0.8, 1.15))
    parser.add_argument('--augmentation_prob', type=float, default=0.1)
    parser.add_argument('--do_GammaTransform', type=helpers.str2bool, default=False)
    parser.add_argument('--do_GaussianNoise', type=helpers.str2bool, default=False)
    parser.add_argument('--brightness', type=tuple, default=(0.7, 1.5))

    # CV
    parser.add_argument('--cv', type=int, default=0)  # cross validation, CV=5
    parser.add_argument('--cv_max', type=int, default=10)

    # Model
    # parser.add_argument('--model', type=str, default="UNET", help="MI/MS/MIMS/UNET")
    parser.add_argument('--n_filter', type=int, default=32)
    #parser.add_argument('--dice_loss', type=float, default=1.0)
    parser.add_argument('--dropout_p', type=float, default=0.8)
    parser.add_argument('--l2', type=float, default=0.)
    parser.add_argument('--dim2', type=helpers.str2bool, default=False)

    args = parser.parse_args()

    #print(args.brightness)
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # load data
    data_gen_tr, data_gen_val, train_patients, val_patients = get_generator(args)
    # train
    if args.dim2 == True:
        print("[x] train 2D segmentation model")
        trainer_2d = ModelTrainer2D(args, sess, input_channel=1, output_channel=1)
        trainer_2d.train(data_gen_tr, data_gen_val, train_patients, val_patients)
    else:
        print("[x] train 3D segmentation model")
        #trainer_3d = ModelTrainer3D_Sampling(args, sess, input_channel=1, output_channel=1)
        trainer_3d = ModelTrainer3D(args, sess, input_channel=1, output_channel=6)
        trainer_3d.train(data_gen_tr, data_gen_val, train_patients, val_patients)
