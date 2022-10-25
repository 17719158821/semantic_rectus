import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import matplotlib.pyplot as plt
import imageio

class DataVisualizer(object):
    def __init__(self, data_for_per_row, save_path):
        self.data = data_for_per_row
        self.save_path = save_path

    def visualize(self, num_per_row=32):
        f, plots = plt.subplots(len(self.data), num_per_row, sharex='all', sharey='all', figsize=(num_per_row, len(self.data)))

        for row in range(len(self.data)):
            for i in range(self.data[row].shape[2]):
                plots[(i + num_per_row * row) // num_per_row, i % num_per_row].axis('off')
                plots[(i + num_per_row * row) // num_per_row, i % num_per_row].imshow(self.data[row][:, :, i], cmap='gray')

        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.01, wspace=0.01)
        plt.margins(0, 0)

        plt.savefig(self.save_path)
        plt.close()


class DataVisualizerGIF(object):

    def __init__(self, data_for_per_row, save_path, patient_name):
        self.data = data_for_per_row
        self.save_path = save_path
        self.patient_name = patient_name

    def generate_gif(self, file_paths, gif_name, duration=0.2):
        frames = []
        for image_name in file_paths:
            frames.append(imageio.imread(image_name))
        imageio.mimsave(gif_name, frames, 'GIF', duration=duration)

    def visualize(self, num_per_row=32):
        image_file_names = []
        for r in range(num_per_row):
            f, plots = plt.subplots(len(self.data), 1, sharex='all', sharey='all', figsize=(1, len(self.data)))
            for row in range(len(self.data)):
                plots[row].axis('off')
                if len(self.data[row].shape)==3:
                    plots[row].imshow(self.data[row][:, :, r])
                else:
                    plots[row].imshow(self.data[row][:, :, r, :])

            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.01, wspace=0.01)
            plt.margins(0, 0)

            plt.savefig(self.save_path+"/%s_%d.png" % (self.patient_name, r))
            image_file_names.append(self.save_path+"/%s_%d.png" % (self.patient_name, r))
            plt.close()

        self.generate_gif(image_file_names, self.save_path+"/%s_gif.gif" % (self.patient_name))

        return image_file_names
