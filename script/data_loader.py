import numpy as np

from time import time
import helpers
import argparse
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.transforms import Compose
from batchgenerators.utilities.data_splitting import get_split_deterministic
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform

import matplotlib.pyplot as plt

"""
def load_data(data_path='./FemurData', image_width=256, image_height=256, cv=0, cv_max=5):
    data_xs = []
    data_ys = []
    patient_dirs = []
    for data_dir in tqdm(os.listdir(data_path)):
        patient_dir = data_path+"/"+data_dir
        #print(patient_dir)
        image_xs = glob(patient_dir+"/data/*.jpg")
        #image_ys = glob(patient_dir+"/data/*.jpg")
        num_image = len(image_xs)
        #print(num_image)
        data_x = np.empty(shape=[image_width, image_height, num_image], dtype=np.float32)
        data_y = np.empty(shape=[image_width, image_height, num_image], dtype=np.float32)
        for i in range(num_image):
            image_x = cv2.imread(patient_dir+"/data/{}.jpg".format(i), cv2.IMREAD_GRAYSCALE)
            image_x = cv2.resize(image_x, (image_width, image_height), interpolation=cv2.INTER_AREA)
            image_y = cv2.imread(patient_dir+"/label/{}.jpg".format(i), cv2.IMREAD_GRAYSCALE)
            image_y = cv2.resize(image_y, (image_width, image_height), interpolation=cv2.INTER_AREA)
            data_x[:, :, i] = image_x /255.
            data_y[:, :, i] = image_y /255.
        patient_dirs.append(patient_dir[patient_dir.rfind('/')+1:])
        data_xs.append(data_x)
        data_ys.append(data_y)

    np.random.seed(12345)
    r_index = np.random.permutation(len(data_xs))
    num_data = len(data_xs)
    num_per_cv = int(num_data / cv_max)

    train_data_x = []
    train_data_y = []
    train_dirs = []
    test_data_x = []
    test_data_y = []
    test_dirs = []

    for i in range(num_data):
        if i not in list(range(cv*num_per_cv, (cv+1)*num_per_cv)):
            train_data_x.append(data_xs[r_index[i]])
            train_data_y.append(data_ys[r_index[i]])
            train_dirs.append(patient_dirs[r_index[i]])
        else:
            test_data_x.append(data_xs[r_index[i]])
            test_data_y.append(data_ys[r_index[i]])
            test_dirs.append(patient_dirs[r_index[i]])

    train_data_x = np.array(train_data_x)
    train_data_y = np.array(train_data_y)
    test_data_x = np.array(test_data_x)
    test_data_y = np.array(test_data_y)
    return train_data_x, train_data_y, train_dirs, test_data_x, test_data_y, test_dirs
"""

def get_list_of_patients(txt_path, sex):
    assert sex in [0, 1, 2]
    patients = []
    if sex == 1:
        f = open(txt_path+"/male_train.txt", "r")
        patients.extend(f.read().splitlines())
        f.close()
    elif sex == 2:
        f = open(txt_path+"/female_train.txt", "r")
        patients.extend(f.read().splitlines())
        f.close()
    else:
        f = open(txt_path + "/female_train.txt", "r")
        patients.extend(f.read().splitlines())
        f.close()
        f = open(txt_path + "/male_train.txt", "r")
        patients.extend(f.read().splitlines())
        f.close()

    return patients


class FemurDataLoader3D(DataLoader):
    def __init__(self,
                 data_path,
                 data,
                 batch_size,
                 patch_size,
                 num_threads_in_multithreaded,
                 seed_for_shuffle=1234,
                 return_incomplete=False,
                 shuffle=True,
                 infinite=True):
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         infinite)

        self.data_path = data_path
        self.patch_size = patch_size
        self.input_channel = 1
        self.output_channel = 1
        self.indices = list(range(len(data)))

    def load_patient(self, patient_id):
        data = np.load("{}/{}_x.npy".format(self.data_path, patient_id), mmap_mode='r')
        seg = np.load("{}/{}_y.npy".format(self.data_path, patient_id), mmap_mode='r')
        data = np.expand_dims(data, axis=0)
        seg = np.expand_dims(seg, axis=0)
        return data, seg

    def generate_train_batch(self):
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]

        # initialize empty array for data and seg
        data = np.zeros((self.batch_size, self.input_channel, *self.patch_size), dtype=np.float32)
        seg = np.zeros((self.batch_size, self.output_channel, *self.patch_size), dtype=np.float32)

        patient_names = []

        for i, j in enumerate(patients_for_batch):
            patient_data, patient_seg = self.load_patient(j)
            patient_data, patient_seg = crop(data=np.expand_dims(patient_data, axis=0), # +b, c, w, h, znp.expand_dims(patient_data, axis=0)
                                             seg=np.expand_dims(patient_seg, axis=0),
                                             crop_size=self.patch_size,
                                             crop_type="random")
            data[i] = patient_data
            seg[i] = patient_seg
            patient_names.append(j)

        # data = np.transpose(data, (0, 2, 3, 4, 1))
        # seg = np.transpose(seg, (0, 2, 3, 4, 1))

        return {'data': data, 'seg': seg, 'names': patient_names}



def get_train_transform(patch_size,args):

    rotation_angle = args.rotation_angle  #15,
    elastic_deform = args.elastic_deform  #(0, 0.25),
    scale_factor = args.scale_factor  #(0.75, 1.25),
    augmentation_prob = args.augmentation_prob  #0.1
    tr_transforms = []

    # the first thing we want to run is the Spatial Transform. It reduces the size of our data to patch_size and thus
    # also reduces the computational cost of all subsequent operations. All subsequent operations do not modify the
    # shape and do not transform spatially, so no border artifacts will be introduced
    # Here we use the new SpatialTransform_2 which uses a new way of parameterizing elastic_deform
    # We use all spatial transformations with a probability of 0.2 per sample. This means that 1 - (1 - 0.1) ** 3 = 27%
    # of samples will be augmented, the rest will just be cropped
    tr_transforms.append(
        SpatialTransform_2(
            patch_size,
            patch_center_dist_from_border=args.patch_center_dist_from_border, #[i // 2 for i in patch_size]
            do_elastic_deform=args.do_elastic_deform, deformation_scale=elastic_deform,
            do_rotation=args.do_rotation,
            angle_x=(0,0), #(- rotation_angle / 360. * 2 * np.pi, rotation_angle / 360. * 2 * np.pi),
            angle_y=(0,0), #(- rotation_angle / 360. * 2 * np.pi, rotation_angle / 360. * 2 * np.pi),
            angle_z=(- rotation_angle / 360. * 2 * np.pi, rotation_angle / 360. * 2 * np.pi),
            do_scale=args.do_scale, scale=scale_factor,
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=1,
            random_crop=args.do_random_crop,
            p_el_per_sample=augmentation_prob,
            p_rot_per_sample=augmentation_prob,
            p_scale_per_sample=augmentation_prob
        )
    )

    # now we mirror along all axes
    #tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))

    # brightness transform for 15% of samples
    tr_transforms.append(BrightnessMultiplicativeTransform(args.brightness,
                                                            per_channel=True,
                                                            p_per_sample=augmentation_prob))

    # gamma transform. This is a nonlinear transformation of intensity values
    # (https://en.wikipedia.org/wiki/Gamma_correction)
    if args.do_GammaTransform ==True:
        tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))
    # we can also invert the image, apply the transform and then invert back
        tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))

    # Gaussian Noise
    if args.do_GaussianNoise == True:
        tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))

    # blurring. Some BraTS cases have very blurry modalities. This can simulate more patients with this problem and
    # thus make the model more robust to it
    #tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,  p_per_channel=0.5, p_per_sample=0.15))

    # new TODO
    #tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.5, 2), per_channel=True, p_per_sample=0.15))
    #tr_transforms.append(ContrastAugmentationTransform(contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True, p_per_sample=0.15))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

def get_filelist_of_patients(path):
    list = []
    for patient_id in sorted(os.listdir(path)):
        list.append(patient_id)
    return list

def get_generator(args):
    """
    obtain data generators for training data and validation data
    :param args:
    :return:
    """
    #patients = get_list_of_patients(args.data_list, sex=args.sex)

    patients = get_filelist_of_patients(args.data_list_path)
    print("[x] found %d patients" % len(patients))
    train_patients, val_patients = get_split_deterministic(patients, fold=args.cv, num_splits=args.cv_max, random_state=12345)
    patch_size = (args.patch_x, args.patch_y, args.patch_z)
    patch_size_load = (args.patch_x, args.patch_y, args.patch_z+4)

    dataloader_train = FemurDataLoader3D(args.data_path, train_patients, args.batch_size, patch_size_load, 1)

    dataloader_validation = FemurDataLoader3D(args.data_path, val_patients, args.batch_size, patch_size, 1)

    tr_transforms = get_train_transform(patch_size,args)

    # use data augmentation shown in tr_transforms to augment training data
    tr_gen = MultiThreadedAugmenter(dataloader_train, tr_transforms,
                                    num_processes=args.n_workers, num_cached_per_queue=5, pin_memory=False)
    # use plain test data without augmentation for testing data
    val_gen = MultiThreadedAugmenter(dataloader_validation, None,
                                     num_processes=args.n_workers, num_cached_per_queue=5, pin_memory=False)

    tr_gen.restart()
    val_gen.restart()
    return tr_gen, val_gen, train_patients, val_patients


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument('--exp', type=str, default="exp_s2_edge")
    parser.add_argument('--num_epochs', type=int, default=1201)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--validate_epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)

    # data
    parser.add_argument('--patch_x', type=int, default=256)
    parser.add_argument('--patch_y', type=int, default=256)
    parser.add_argument('--patch_z', type=int, default=16)
    # parser.add_argument('--data_list_path', type=str, default='./data_uci/all/label') # path for female.txt ...
    parser.add_argument('--data_list_path', type=str, default='./data_uci/label')  # path for female.txt ...
    parser.add_argument('--data_path', type=str, default="./data_uci/data_uci_converted")
    parser.add_argument('--sex', type=int, default=0, help="0 for all, 1 for male, 2 for female")

    # cpu
    parser.add_argument('--n_workers', type=int, default=6)

    # gpu
    parser.add_argument('--gpu', type=str, default="0")

    # data augmentation
    parser.add_argument('--patch_center_dist_from_border', type=tuple, default=(256//2-5, 256//2-5,16//2))
    parser.add_argument('--do_random_crop', type=helpers.str2bool, default=True)
    parser.add_argument('--do_rotation', type=helpers.str2bool, default=True)
    parser.add_argument('--rotation_angle',type=int,default=10)
    parser.add_argument('--do_elastic_deform', type=helpers.str2bool, default=False)
    parser.add_argument('--elastic_deform', type=tuple, default=(0, 0.05))
    parser.add_argument('--do_scale', type=helpers.str2bool, default=False)
    parser.add_argument('--scale_factor', type=tuple, default=(0.8, 1.15))
    parser.add_argument('--augmentation_prob', type=float, default=1)
    parser.add_argument('--do_GammaTransform', type=helpers.str2bool, default=False)
    parser.add_argument('--do_GaussianNoise', type=helpers.str2bool, default=False)
    parser.add_argument('--brightness', type=tuple, default=(0.1, 0.2))

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

    #patients = get_list_of_patients('./data_uci', sex=1)
    patients = get_filelist_of_patients('./data_uci/all/label')
    print(len(patients))

    train_patients, val_patients = get_split_deterministic(patients, fold=0, num_splits=5, random_state=12345)

    patch_size_load = (256, 256, 22)
    patch_size = (256, 256, 16)
    batch_size = 1

    data_path = './data_uci/data_uci_converted'
    num_threads_in_multithreaded = 2

    dataloader = FemurDataLoader3D(data_path, train_patients, batch_size, patch_size_load, num_threads_in_multithreaded)

    batch = next(dataloader)
    print(batch)
                       #FemurDataLoader3D(data_path, train_patients, batch_size, patch_size, num_threads_in_multithreaded)
    dataloader_train = FemurDataLoader3D(data_path, train_patients, batch_size, patch_size_load, 1)

    dataloader_validation = FemurDataLoader3D(data_path, val_patients, batch_size, patch_size, 1)

    tr_transforms = get_train_transform(patch_size,args)

    tr_gen = MultiThreadedAugmenter(dataloader_train, tr_transforms,
                                    num_processes=num_threads_in_multithreaded,
                                    num_cached_per_queue=3,
                                    pin_memory=False)
    val_gen = MultiThreadedAugmenter(dataloader_validation, None,
                                     num_processes=num_threads_in_multithreaded,
                                     num_cached_per_queue=3,
                                     pin_memory=False)

    tr_gen.restart()
    val_gen.restart()

    num_batches_per_epoch = 100
    num_validation_batches_per_epoch = 3
    num_epochs = 5
    # let's run this to get a time on how long it takes
    time_per_epoch = []
    start = time()
    for epoch in range(num_epochs):
        start_epoch = time()
        for b in range(num_batches_per_epoch):
            batch = next(tr_gen)
            imgshow_data = np.squeeze(batch["data"]).transpose((2,0,1))
            imgshow_label = np.squeeze(batch["seg"]).transpose((2,0,1))
            for i in range(0,32,2):
                savepath_data = "C:/Users/G/Desktop/gzy/gzy/data_uci/aug/{}.png".format(i)
                savepath_label = "C:/Users/G/Desktop/gzy/gzy/data_uci/aug/{}.png".format(i+1)
                plt.imsave(savepath_data, imgshow_data[i//2])
                plt.imsave(savepath_label, imgshow_label[i//2])

            # do network training here with this batch

        for b in range(num_validation_batches_per_epoch):
            batch = next(val_gen)
            # run validation here
        end_epoch = time()
        time_per_epoch.append(end_epoch - start_epoch)
    end = time()
    total_time = end - start
    print("Running %d epochs took a total of %.2f seconds with time per epoch being %s" %
          (num_epochs, total_time, str(time_per_epoch)))

