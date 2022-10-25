import numpy as np
import os
import argparse
import cv2
import matplotlib
import matplotlib.pyplot as plt
import imageio

from glob import glob
from skimage import measure
from visualize import DataVisualizer, DataVisualizerGIF
from tqdm import tqdm

IMAGE_INCH = 1.92
#IMAGE_INCH=5


def convert_img_to_np(image_path, image_size=(192, 192), gray=False):
    if gray:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, image_size)
    return img


def skeleton_3D_image(bin_images, gt_images, original_images, patient_name, save_path,
                      line_colors=['r', 'g'], linewidth=1):
    assert len(np.shape(bin_images)) == 3  # x, y, z for width, height, len
    skeleton_images = np.zeros(list(np.shape(bin_images))+[3])
    skeleton_image_paths = []
    for i in range(bin_images.shape[2]):
        bin_contours = measure.find_contours(bin_images[:, :, i], 0.5)
        gt_contours = measure.find_contours(gt_images[:, :, i], 0.5)

        fig, ax = plt.subplots()
        ax.imshow(original_images[:, :, i], cmap=plt.cm.gray)
        # gt is red
        for n, contour in enumerate(gt_contours):
            ax.plot(contour[:, 1], contour[:, 0], line_colors[0], linewidth=linewidth)

        # bin is green
        for n, contour in enumerate(bin_contours):
            ax.plot(contour[:, 1], contour[:, 0], line_colors[1], linewidth=linewidth)

        plt.axis('off')
        fig.set_size_inches(IMAGE_INCH, IMAGE_INCH)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
        plt.margins(0, 0)
        plt.savefig(fname="%s/%s/%s_%d_contour.png" % (save_path, patient_name, patient_name, i), pad_inches=0.)
        plt.close()
        skeleton_image = convert_img_to_np("./tmp.png", (skeleton_images.shape[0], skeleton_images.shape[1]))
        skeleton_images[:, :, i, :] = skeleton_image
        skeleton_image_paths.append("%s/%s/%s_%d_contour.png" % (save_path, patient_name, patient_name, i))
    return skeleton_images, skeleton_image_paths


def overlay_3D_image(overlay_images, original_images, patient_name, save_path, image_pattern="gt", fill_color='g', draw_ori=True):
    assert len(np.shape(overlay_images)) == 3  # x, y, z for width, height, len
    render_image_paths = []

    for i in range(overlay_images.shape[2]):
        overlay_contours = measure.find_contours(overlay_images[:, :, i], 0.5)
        fig, ax = plt.subplots()
        #fig, ax = plt.subplots()
        if draw_ori:
            ax.imshow(original_images[:, :, i], cmap=plt.cm.gray)
        else:
            ax.imshow(np.zeros_like(original_images[:, :, i]), cmap=plt.cm.gray)
        # gt is red
        for n, contour in enumerate(overlay_contours):
            ax.fill(contour[:, 1], contour[:, 0], fill_color, alpha=1)

        plt.axis('off')
        fig.set_size_inches(IMAGE_INCH, IMAGE_INCH)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
        plt.margins(0, 0)
        plt.savefig(fname="%s/%s/%s_%d_%s_overlay.png" % (save_path, patient_name, patient_name, i, image_pattern), pad_inches=0., dpi=100)
        plt.close()
        render_image_paths.append("%s/%s/%s_%d_%s_overlay.png" % (save_path, patient_name, patient_name, i, image_pattern))

    return render_image_paths


def original_3D_image(original_images, patient_name, save_path):
    assert len(np.shape(original_images)) == 3
    original_image_paths = []
    for i in range(original_images.shape[2]):
        fig, ax = plt.subplots()
        ax.imshow(original_images[:, :, i], cmap=plt.cm.gray)
        plt.axis('off')
        fig.set_size_inches(IMAGE_INCH, IMAGE_INCH)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
        plt.margins(0, 0)
        plt.savefig(fname="%s/%s/%s_%d_x.png" % (save_path, patient_name, patient_name, i), pad_inches=0.)
        plt.close()
        original_image_paths.append("%s/%s/%s_%d_x.png" % (save_path, patient_name, patient_name, i))

    return original_image_paths


def generate_gif(file_paths, gif_name, duration=1.0):
    frames = []
    for image_name in file_paths:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)


def convert_overlay_path_to_np(file_paths, image_size=192, color_value=75):
    images = np.zeros([image_size, image_size, len(file_paths)], dtype=np.int32)
    for idx, file_path in enumerate(file_paths):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img_binary = np.zeros([image_size, image_size])
        img_binary[img==color_value] = 1
        """
        for i in range(image_size):
            for j in range(image_size):
                if img[i, j] == color_value:
                    img_binary[i, j] = 1
        """
        images[:, :, idx] = img_binary
    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiment    parser.add_argument('--epoch_dir', type=str, default="exp_s0/0099")
    #parser.add_argument('--epoch_dir', type=str, default="exp_s0/0099")
    parser.add_argument('--epoch_dir', type=str, default="exp_s2_edge/0810")
    # parser.add_argument('--epoch_dir', type=str, default="exp_s2/0064")
    args = parser.parse_args()

    epoch_dir = args.epoch_dir

    bin_npys = sorted(glob(epoch_dir + "/*_bin.npy"))
    gt_npys = sorted(glob(epoch_dir + "/*_gt.npy"))
    x_npys = sorted(glob(epoch_dir + "/*_x.npy"))
    for i in tqdm(range(len(bin_npys))):
        patient_name = bin_npys[i][bin_npys[i].rfind("/")+1: bin_npys[i].rfind("_bin")]
        if not os.path.isdir("%s/%s" % (epoch_dir, patient_name)):
            os.makedirs("%s/%s" % (epoch_dir, patient_name))
        if not os.path.isdir("%s/%s" % (epoch_dir, patient_name+"_fake")):
            os.makedirs("%s/%s" % (epoch_dir, patient_name+"_fake"))

        x_paths = original_3D_image(original_images=np.squeeze(np.load(x_npys[i])),
                                    patient_name=patient_name,
                                    save_path=epoch_dir)

        # gt overlay
        overlay_gt_paths = overlay_3D_image(overlay_images=np.squeeze(np.load(gt_npys[i])),
                                            original_images=np.squeeze(np.load(x_npys[i])),
                                            patient_name=patient_name,
                                            save_path=epoch_dir,
                                            image_pattern='gt',
                                            fill_color='r')
        # bin overlay
        overlay_bin_paths = overlay_3D_image(overlay_images=np.squeeze(np.load(bin_npys[i])),
                                             original_images=np.squeeze(np.load(x_npys[i])),
                                             patient_name=patient_name,
                                             save_path=epoch_dir,
                                             image_pattern='bin',
                                             fill_color='g')

        overlay_bin_paths = overlay_3D_image(overlay_images=np.squeeze(np.load(bin_npys[i])),
                                             original_images=np.squeeze(np.load(x_npys[i])),
                                             patient_name=patient_name+"_fake",
                                             save_path=epoch_dir,
                                             image_pattern='bin',
                                             fill_color='g',
                                             draw_ori=False)

        overlay_bin_images = convert_overlay_path_to_np(overlay_bin_paths)

        skeleton_npy, skeleton_image_paths = skeleton_3D_image(bin_images=overlay_bin_images,
                                                               gt_images=np.squeeze(np.load(gt_npys[i])),
                                                               original_images=np.squeeze(np.load(x_npys[i])),
                                                               patient_name=patient_name,
                                                               save_path=epoch_dir,
                                                               line_colors=['r', 'g']) # red for gt, green for bin
        #generate_gif(skeleton_image_paths, gif_name="%s/%s_gif.gif" % (epoch_dir, patient_name))

        #generate_gif(overlay_bin_paths, gif_name="%s/%s_bin_overlay_gif.gif" % (epoch_dir, patient_name))


        """
        skeleton_3D_image2(bin_images=np.squeeze(np.load(bin_npys[i])),
                           gt_images=np.squeeze(np.load(gt_npys[i])),
                           original_images=np.squeeze(np.load(x_npys[i])),
                           patient_name=patient_name,
                           save_path=epoch_dir,
                           line_colors=['r', 'g'])
        """

        # generate_gif(overlay_gt_paths, gif_name="%s/%s_gt_overlay_gif.gif" % (epoch_dir, patient_name))


        # generate_gif(x_paths, gif_name="%s/%s_x_gif.gif" % (epoch_dir, patient_name))

        """
        dv_gif = DataVisualizerGIF([np.squeeze(np.load(x_npys[i])),
                                    np.squeeze(np.load(bin_npys[i])),
                                    np.squeeze(skeleton_npy_bin),
                                    np.squeeze(np.load(gt_npys[i])),
                                    np.squeeze(skeleton_npy_gt)],
                                   save_path=os.path.join(args.exp, epoch_dir),
                                   patient_name=patient_name)
        
        dv_gif = DataVisualizerGIF([np.squeeze(np.load(bin_npys[i])),
                                    skeleton_npy,
                                    np.squeeze(np.load(gt_npys[i]))],
                                   save_path=os.path.join(args.exp, epoch_dir),
                                   patient_name=patient_name)
        dv_gif.visualize(num_per_row=np.squeeze(np.load(bin_npys[i])).shape[2])

        dv = DataVisualizer([np.squeeze(np.load(bin_npys[i])),
                             np.squeeze(skeleton_npy_bin),
                             np.squeeze(np.load(gt_npys[i])),
                             np.squeeze(skeleton_npy_gt)],
                            save_path=os.path.join(args.exp, epoch_dir) + "/%s_contours.png" % (patient_name))
        dv.visualize(num_per_row=np.squeeze(np.load(bin_npys[i])).shape[2])
        """