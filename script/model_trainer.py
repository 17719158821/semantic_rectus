import tensorflow as tf
import os
import numpy as np

from layers import conv_bn_relu_drop, resnet_Add, down_sampling, deconv_relu, crop_and_concat, conv_sigmod
from tqdm import tqdm

from evaluator import evaluate_single_image
from visualize import DataVisualizer
from models import VNet, MSUNet, MobileUNet, count_params
from make_edges import overlay_3D_image
from post_treatment import post_treatment
from DICE_evaluate import dice_evaluate
from utils import save_result
# from keras.utils import to_categorical
from tensorflow.python.keras.utils import to_categorical

class ModelTrainer3D(object):

    def __init__(self, args, sess, input_channel, output_channel):
        self.args = args
        self.input_channel = input_channel
        self.output_channel = output_channel

        self.x = tf.placeholder('float', shape=[None, self.args.patch_x, self.args.patch_y, self.args.patch_z, self.input_channel])
        self.y_gt = tf.placeholder('float', shape=[None, self.args.patch_x, self.args.patch_y, self.args.patch_z, self.output_channel])
        self.lr = tf.placeholder('float')
        self.drop = tf.placeholder('float')

        vnet = VNet(self.args.n_filter, self.x, self.output_channel, self.drop, self.args.l2)
        self.y_pred = vnet.create_model()
        self.cost = self.get_cost()
        self.sess = sess
        count_params()

    def get_cost(self, dice_loss=False):

        if dice_loss:
            # dice
            Z, H, W, C = self.y_gt.get_shape().as_list()[1:]
            smooth = 1e-5
            pred_flat = tf.reshape(self.y_pred, [-1, H * W * C * Z])
            true_flat = tf.reshape(self.y_gt, [-1, H * W * C * Z])
            intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
            denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
            loss = -tf.reduce_mean(intersection / denominator)
            return loss
        else:
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_pred, labels=self.y_gt)
            loss = tf.reduce_mean(losses)
            return loss

    def train(self, data_generator_train, data_generator_validation, train_patients, val_patients):
        num_train_data = len(train_patients)
        num_val_data = len(val_patients)
        train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        init = tf.global_variables_initializer()
        global_evaluation = 0
        saver = tf.train.Saver(max_to_keep=2000)
        self.sess.run(init)

        for epoch in tqdm(range(self.args.num_epochs)):
            print("[x] epoch: %d, training" % epoch)
            num_batch =  num_train_data//self.args.batch_size
            epoch_loss = 0.
            for mini_batch in tqdm(range(num_batch)):
                batch_data = next(data_generator_train)
                _, train_loss = self.sess.run([train_op, self.cost],
                                              feed_dict={self.x: np.transpose(batch_data['data'], (0, 2, 3, 4, 1)),
                                                         self.y_gt: np.expand_dims(to_categorical(np.squeeze(batch_data['seg']),num_classes=6),axis=0),
                                                         self.lr: self.args.lr,
                                                         self.drop: self.args.dropout_p})
                epoch_loss += train_loss

            print("[x] epoch: %d, loss: %f" % (epoch, epoch_loss/num_batch))
            if (epoch) % self.args.validate_epoch == 0:
                if not os.path.isdir(self.args.exp + "/%04d" % (epoch)):
                    os.makedirs(self.args.exp + "/%04d" % (epoch))
                print("[x] epoch: %d, validate" % epoch)
                Dice_result = []
                model_results = []
                # Dice_result_L = []
                for mini_batch in tqdm(range(num_val_data)):
                    """
                    batch_data = next(data_generator_validation)
                    pred_y = self.predict(batch_data)
                    model_result = self.evaluate_results(np.transpose(batch_data['seg'], (0, 2, 3, 4, 1)),
                                                         pred_y,
                                                         batch_data['names'],
                                                         np.transpose(batch_data['data'], (0, 2, 3, 4, 1)))
                    model_results.extend(model_result)
                    """
                    # model_results_largest = []
                    patient_id = val_patients[mini_batch]
                    data_x = np.load("{}/{}_x.npy".format(self.args.data_path, patient_id), mmap_mode='r')
                    data_y = np.load("{}/{}_y.npy".format(self.args.data_path, patient_id), mmap_mode='r')
                    data_x = np.expand_dims(data_x, axis=3)
                    data_x = np.expand_dims(data_x, axis=0)
                    data_y = np.expand_dims(to_categorical(data_y, num_classes=6), axis=0)

                    pred_y = self.inference_whole_volume(data_x, data_y.shape)

                    model_result,dice_eval = dice_evaluate(pred_y,data_y,data_x,[patient_id])
                    model_results.extend(model_result)
                    Dice_result.append(dice_eval)
                #     pred_y_largest = np.expand_dims(post_treatment(pred_y_bin), axis=0)
                #     model_result_largest,_ = self.evaluate_results(data_y,
                #                                           pred_y_largest,
                #                                           [patient_id],
                #                                           data_x)
                #     model_results_largest.extend(model_result_largest)
                #     Dice_result_L.append(model_result_largest[0]['dice'])
                save_result(model_results,self.args.exp,epoch)
                # self.save_result(model_results_largest, epoch, L=True)
                print("Average_DICE:{}".format(np.mean(Dice_result)))
                if np.mean(Dice_result) > global_evaluation and epoch > 0:
                    model_checkpoint_name = "{}/latest_model.ckpt".format(self.args.exp)
                    print("[x] saving model to{}".format(model_checkpoint_name))
                    global_evaluation = np.mean(Dice_result)
                    saver.save(self.sess, model_checkpoint_name)
               

    def predict(self, batch_data):
        batch_pred = self.sess.run(self.y_pred, feed_dict={self.x: np.transpose(batch_data['data'], (0, 2, 3, 4, 1)),
                                                           self.drop: 1.})
        return batch_pred

    def inference_whole_volume(self, data,shape, interval_z=1):
        """
        inference segmentation results on whole slices
        :param data: shape [1, patch_x, patch_y, patch_z, channel]
        :return:
        """
        inferenced_volume = np.zeros(shape).astype('float32')
        inferenced_time = np.zeros(shape).astype('float32')

        for slice_z in range(0, data.shape[3] - self.args.patch_z + 1, interval_z):
            part_pred = self.sess.run(self.y_pred, feed_dict={self.x: data[:, :, :, slice_z: slice_z+self.args.patch_z, :], self.drop: 1.})
            inferenced_volume[:, :, :, slice_z: slice_z+self.args.patch_z:, :] += part_pred
            inferenced_time[:, :, :, slice_z: slice_z+self.args.patch_z:, :] += 1
        inferenced_volume = inferenced_volume/inferenced_time
        return inferenced_volume

    def evaluate_results(self, gt, y_pred, patient_ids, x):
        evaluation_list = []
        for i in range(gt.shape[0]):
            binary_image, auc_roc, auc_pr, dice_coeff, acc, sensitivity, specificity = \
                evaluate_single_image(y_pred[i], gt[i])
            result = {'patient_id': patient_ids[i],
                      'bin': binary_image,
                      'pred': y_pred[i],
                      'x': x[i],
                      'gt': gt[i],
                      'acc': acc,
                      'sn': sensitivity,
                      'sp': specificity,
                      'dice': dice_coeff,
                      'auc_roc': auc_roc,
                      'auc_pr': auc_pr}
            evaluation_list.append(result)
        return evaluation_list,list(binary_image)

    def save_result(self, evaluated_results, epoch, L):
        if not os.path.isdir(self.args.exp + "/%04d" % (epoch)):
            os.makedirs(self.args.exp + "/%04d" % (epoch))

        if L == False:
            target = open(self.args.exp + "/%04d/val.csv" % (epoch), "w")
        else:
            target = open(self.args.exp + "/%04d/val_L.csv" % (epoch), "w")
        target.write('patient,auc_roc,auc_pr,dice,acc,sn,sp\n')

        for i, evaluated_result in enumerate(evaluated_results):
            target.write("%s,%f,%f,%f,%f,%f,%f\n" % (evaluated_result['patient_id'],
                                                     evaluated_result['auc_roc'],
                                                     evaluated_result['auc_pr'],
                                                     evaluated_result['dice'],
                                                     evaluated_result['acc'],
                                                     evaluated_result['sn'],
                                                     evaluated_result['sp']))
            if L == False:
                np.save(file=self.args.exp + "/%04d/%s_x.npy" % (epoch, evaluated_result['patient_id']),
                        arr=evaluated_result['x'])
                np.save(file=self.args.exp + "/%04d/%s_pred.npy" % (epoch, evaluated_result['patient_id']),
                        arr=evaluated_result['pred'])
                np.save(file=self.args.exp + "/%04d/%s_gt.npy" % (epoch, evaluated_result['patient_id']),
                        arr=evaluated_result['gt'])
                np.save(file=self.args.exp + "/%04d/%s_bin.npy" % (epoch, evaluated_result['patient_id']),
                        arr=evaluated_result['bin'])
                dv = DataVisualizer([np.squeeze(evaluated_result['x']),
                                     np.squeeze(evaluated_result['bin']),
                                     np.squeeze(evaluated_result['pred']),
                                     np.squeeze(evaluated_result['gt'])],
                                    save_path=self.args.exp + "/%04d/%s.png" % (epoch, evaluated_result['patient_id']))
            else:
                np.save(file=self.args.exp + "/%04d/%s_bin_L.npy" % (epoch, evaluated_result['patient_id']),
                        arr=evaluated_result['bin'])
                dv = DataVisualizer([np.squeeze(evaluated_result['x']),
                                     np.squeeze(evaluated_result['bin']),
                                     np.squeeze(evaluated_result['pred']),
                                     np.squeeze(evaluated_result['gt'])],
                                    save_path=self.args.exp + "/%04d/%s_L.png" % (epoch, evaluated_result['patient_id']))
            dv.visualize(evaluated_result['x'].shape[2])
        target.close()


class ModelTrainer3D_Sampling(object):

    def __init__(self, args, sess, input_channel, output_channel):
        self.args = args
        self.input_channel = input_channel
        self.output_channel = output_channel

        self.x = tf.placeholder('float', shape=[None, self.args.patch_x, self.args.patch_y, self.args.patch_z, self.input_channel])
        self.y_gt = tf.placeholder('float', shape=[None, self.args.patch_x, self.args.patch_y, self.args.patch_z, self.output_channel])
        self.lr = tf.placeholder('float')
        self.drop = tf.placeholder('float')

        vnet = VNet(self.args.n_filter, self.x, self.output_channel, self.drop, self.args.l2)
        self.y_pred = vnet.create_model()
        self.cost = self.get_cost()
        self.sess = sess
        count_params()

    def get_cost(self, dice_loss=False):
        if dice_loss:
            # dice
            Z, H, W, C = self.y_gt.get_shape().as_list()[1:]
            smooth = 1e-5
            pred_flat = tf.reshape(self.y_pred, [-1, H * W * C * Z])
            true_flat = tf.reshape(self.y_gt, [-1, H * W * C * Z])
            intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
            denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
            loss = -tf.reduce_mean(intersection / denominator)
            return loss
        else:
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.y_gt)
            loss = tf.reduce_mean(losses)
            return loss

    def train(self, data_generator_train, data_generator_validation, train_patients, val_patients):
        num_train_data = len(train_patients)
        num_val_data = len(val_patients)
        train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        init = tf.global_variables_initializer()

        self.sess.run(init)

        for epoch in tqdm(range(self.args.num_epochs)):
            print("[x] epoch: %d, training" % epoch)
            num_batch = num_train_data//self.args.batch_size
            epoch_loss = 0.
            for mini_batch in range(num_batch):
                batch_data = next(data_generator_train)
                # calculate cost

                pred_volume = self.sess.run(self.y_pred,
                                            feed_dict={self.x: np.transpose(batch_data['data'], (0, 2, 3, 4, 1)),
                                                       self.drop: 1.})

                # calculate dice coef for each images
                dices = []
                for i in range(self.args.patch_z):  # self.args.patch_z
                    pred_image = pred_volume[:, :, :, i, :]
                    gt_image = np.transpose(batch_data['seg'], (0, 2, 3, 4, 1))[:, :, :, i, :]
                    binary_image, auc_roc, auc_pr, dice_coeff, acc, sensitivity, specificity = \
                        evaluate_single_image(pred_image, gt_image)
                    dices.append(dice_coeff)

                # sampling
                dice_coeff = np.array(dice_coeff)
                dice_min_index = np.argmin(dice_coeff)
                train_x = np.transpose(batch_data['data'], (0, 2, 3, 4, 1))
                if dice_min_index < self.args.patch_z - 2:
                    train_x[:, :, :, dice_min_index+1, :] = train_x[:, :, :, dice_min_index, :]

                # train
                _, train_loss = self.sess.run([train_op, self.cost],
                                              feed_dict={self.x: np.transpose(batch_data['data'], (0, 2, 3, 4, 1)),
                                                         self.y_gt: np.transpose(batch_data['seg'], (0, 2, 3, 4, 1)),
                                                         self.lr: self.args.lr,
                                                         self.drop: self.args.dropout_p})
                epoch_loss += train_loss

            print("[x] epoch: %d, loss: %f" % (epoch, epoch_loss/num_batch))
            if (epoch) % self.args.validate_epoch == 0:
                print("[x] epoch: %d, validate" % epoch)
                model_results = []
                for mini_batch in range(num_val_data):
                    """
                    batch_data = next(data_generator_validation)
                    pred_y = self.predict(batch_data)
                    model_result = self.evaluate_results(np.transpose(batch_data['seg'], (0, 2, 3, 4, 1)),
                                                         pred_y,
                                                         batch_data['names'],
                                                         np.transpose(batch_data['data'], (0, 2, 3, 4, 1)))
                    model_results.extend(model_result)
                    """
                    patient_id = val_patients[mini_batch]
                    data_x = np.load("{}/{}_x.npy".format(self.args.data_path, patient_id), mmap_mode='r')
                    data_y = np.load("{}/{}_y.npy".format(self.args.data_path, patient_id), mmap_mode='r')
                    data_x = np.expand_dims(data_x, axis=3)
                    data_x = np.expand_dims(data_x, axis=0)
                    data_y = np.expand_dims(data_y, axis=3)
                    data_y = np.expand_dims(data_y, axis=0)

                    pred_y = self.inference_whole_volume(data_x)

                    model_result = self.evaluate_results(data_y,
                                                         pred_y,
                                                         [patient_id],
                                                         data_x)
                    model_results.extend(model_result)

                self.save_result(model_results, epoch)

    def predict(self, batch_data):
        batch_pred = self.sess.run(self.y_pred, feed_dict={self.x: np.transpose(batch_data['data'], (0, 2, 3, 4, 1)),
                                                           self.drop: 1.})
        return batch_pred

    def inference_whole_volume(self, data, interval_z=1):
        """
        inference segmentation results on whole slices
        :param data: shape [1, patch_x, patch_y, patch_z, channel]
        :return:
        """
        inferenced_volume = np.zeros_like(data)
        inferenced_time = np.zeros_like(data)
        for slice_z in range(0, data.shape[3] - self.args.patch_z + 1, interval_z):
            part_pred = self.sess.run(self.y_pred, feed_dict={self.x: data[:, :, :, slice_z: slice_z+self.args.patch_z, :], self.drop: 1.})
            inferenced_volume[:, :, :, slice_z: slice_z+self.args.patch_z:, :] += part_pred
            inferenced_time[:, :, :, slice_z: slice_z+self.args.patch_z:, :] += 1
        inferenced_volume = inferenced_volume/inferenced_time
        return inferenced_volume

    def evaluate_results(self, gt, y_pred, patient_ids, x):
        evaluation_list = []
        for i in range(self.args.batch_size):
            binary_image, auc_roc, auc_pr, dice_coeff, acc, sensitivity, specificity = \
                evaluate_single_image(y_pred[i], gt[i])
            result = {'patient_id': patient_ids[i],
                      'bin': binary_image,
                      'pred': y_pred[i],
                      'x': x[i],
                      'gt': gt[i],
                      'acc': acc,
                      'sn': sensitivity,
                      'sp': specificity,
                      'dice': dice_coeff,
                      'auc_roc': auc_roc,
                      'auc_pr': auc_pr}
            evaluation_list.append(result)
        return evaluation_list

    def save_result(self, evaluated_results, epoch):
        if not os.path.isdir(self.args.exp + "/%04d" % (epoch)):
            os.makedirs(self.args.exp + "/%04d" % (epoch))

        target = open(self.args.exp + "/%04d/val.csv" % (epoch), "w")
        target.write('patient,auc_roc,auc_pr,dice,acc,sn,sp\n')

        for i, evaluated_result in enumerate(evaluated_results):
            target.write("%s,%f,%f,%f,%f,%f,%f\n" % (evaluated_result['patient_id'],
                                                     evaluated_result['auc_roc'],
                                                     evaluated_result['auc_pr'],
                                                     evaluated_result['dice'],
                                                     evaluated_result['acc'],
                                                     evaluated_result['sn'],
                                                     evaluated_result['sp']))

            np.save(file=self.args.exp + "/%04d/%s_x.npy" % (epoch, evaluated_result['patient_id']),
                    arr=evaluated_result['x'])
            np.save(file=self.args.exp + "/%04d/%s_bin.npy" % (epoch, evaluated_result['patient_id']),
                    arr=evaluated_result['bin'])
            np.save(file=self.args.exp + "/%04d/%s_pred.npy" % (epoch, evaluated_result['patient_id']),
                    arr=evaluated_result['pred'])
            np.save(file=self.args.exp + "/%04d/%s_gt.npy" % (epoch, evaluated_result['patient_id']),
                    arr=evaluated_result['gt'])

            dv = DataVisualizer([np.squeeze(evaluated_result['x']),
                                 np.squeeze(evaluated_result['bin']),
                                 np.squeeze(evaluated_result['pred']),
                                 np.squeeze(evaluated_result['gt'])],
                                 save_path=self.args.exp + "/%04d/%s.png" % (epoch, evaluated_result['patient_id']))
            dv.visualize(evaluated_result['x'].shape[2])

        target.close()


class ModelTrainer2D(object):
    def __init__(self, args, sess, input_channel, output_channel):
        self.args = args
        self.input_channel = input_channel
        self.output_channel = output_channel

        self.x = tf.placeholder('float', shape=[None, self.args.patch_x, self.args.patch_y, self.input_channel])
        self.y_gt = tf.placeholder('float', shape=[None, self.args.patch_x, self.args.patch_y, self.output_channel])
        self.lr = tf.placeholder('float')
        self.drop = tf.placeholder('float')

        #vnet = VNet(self.args.n_filter, self.x, self.output_channel, self.drop, self.args.l2)
        msunet = MSUNet(self.args.n_filter, self.x, self.output_channel, self.drop, self.args.l2)
        #unet = MobileUNet(self.args.n_filter, self.x, self.output_channel, self.drop, self.args.l2)
        self.y_pred = msunet.create_model()
        self.cost = self.get_cost()
        self.sess = sess

    def get_cost(self):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.y_gt)
        loss = tf.reduce_mean(losses)

        return loss

    def train(self, data_generator_train, data_generator_validation, train_patients, val_patients):
        num_train_data = len(train_patients)
        num_val_data = len(val_patients)
        train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        init = tf.global_variables_initializer()

        self.sess.run(init)

        for epoch in tqdm(range(self.args.num_epochs)):
            print("[x] epoch: %d, training" % epoch)
            #num_batch = num_train_data//self.args.batch_size
            for i in tqdm(range(num_train_data)):
                batch_data = next(data_generator_train)
                batch_xs = np.transpose(batch_data['data'], (0, 2, 3, 4, 1))
                batch_ys = np.transpose(batch_data['seg'], (0, 2, 3, 4, 1))
                for mini_batch in range(batch_xs.shape[3]):
                    _, train_loss = self.sess.run([train_op, self.cost],
                                                  feed_dict={self.x: batch_xs[:, :, :, mini_batch],  # to (1, patch_x, patch_y, 1)
                                                             self.y_gt: batch_ys[:, :, :, mini_batch],  # to (1, patch_x, patch_y, 1)
                                                             self.lr: self.args.lr,
                                                             self.drop: self.args.dropout_p})

            if (epoch+1) % self.args.validate_epoch == 0:
                print("[x] epoch: %d, validate" % epoch)

                model_results = []

                for val_data_idx in tqdm(range(num_val_data)):
                    batch_data = next(data_generator_validation)
                    pred_y = self.predict(batch_data)
                    model_result = self.evaluate_results(np.transpose(batch_data['seg'], (0, 2, 3, 4, 1)),
                                                         pred_y,
                                                         batch_data['names'],
                                                         np.transpose(batch_data['data'], (0, 2, 3, 4, 1)))
                    model_results.extend(model_result)

                self.save_result(model_results, epoch)

    def predict(self, batch_data):
        batch_pred = np.zeros_like(np.transpose(batch_data['seg'], (0, 2, 3, 4, 1)))
        batch_xs = np.transpose(batch_data['data'], (0, 2, 3, 4, 1))
        for mini_batch in range(batch_xs.shape[3]):
            batch_pred[:, :, :, mini_batch, :] = self.sess.run(self.y_pred,
                                                               feed_dict={self.x: batch_xs[:, :, :, mini_batch],
                                                                          self.drop: 1.})
        return batch_pred

    def evaluate_results(self, gt, y_pred, patient_ids, x):
        evaluation_list = []
        for i in range(self.args.batch_size):
            binary_image, auc_roc, auc_pr, dice_coeff, acc, sensitivity, specificity = \
                evaluate_single_image(y_pred[i], gt[i])
            result = {'patient_id': patient_ids[i],
                      'bin': binary_image,
                      'pred': y_pred[i],
                      'x': x[i],
                      'gt': gt[i],
                      'acc': acc, 'sn': sensitivity, 'sp': specificity, 'dice': dice_coeff,
                      'auc_roc': auc_roc, 'auc_pr': auc_pr}
            evaluation_list.append(result)
        return evaluation_list

    def save_result(self, evaluated_results, epoch):
        if not os.path.isdir(self.args.exp + "/%04d" % (epoch)):
            os.makedirs(self.args.exp + "/%04d" % (epoch))

        target = open(self.args.exp + "/%04d/val.csv" % (epoch), "w")
        target.write('patient,auc_roc,auc_pr,dice,acc,sn,sp\n')

        for i, evaluated_result in enumerate(evaluated_results):
            target.write("%s,%f,%f,%f,%f,%f,%f\n" % (evaluated_result['patient_id'],
                                                     evaluated_result['auc_roc'],
                                                     evaluated_result['auc_pr'],
                                                     evaluated_result['dice'],
                                                     evaluated_result['acc'],
                                                     evaluated_result['sn'],
                                                     evaluated_result['sp']))
            np.save(file=self.args.exp+"/%04d/%s_x.npy" % (epoch, evaluated_result['patient_id']),
                    arr=evaluated_result['x'])
            np.save(file=self.args.exp+"/%04d/%s_bin.npy" % (epoch, evaluated_result['patient_id']),
                     arr=evaluated_result['bin'])
            np.save(file=self.args.exp + "/%04d/%s_pred.npy" % (epoch, evaluated_result['patient_id']),
                    arr=evaluated_result['pred'])
            np.save(file=self.args.exp + "/%04d/%s_gt.npy" % (epoch, evaluated_result['patient_id']),
                    arr=evaluated_result['gt'])

            dv = DataVisualizer([np.squeeze(evaluated_result['x']),
                                 np.squeeze(evaluated_result['bin']),
                                 np.squeeze(evaluated_result['pred']),
                                 np.squeeze(evaluated_result['gt'])],
                                save_path=self.args.exp + "/%04d/%s.png" % (epoch, evaluated_result['patient_id']))
            dv.visualize(num_per_row=evaluated_result['x'].shape[2])

        target.close()
