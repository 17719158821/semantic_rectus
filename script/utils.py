import numpy as np


from script.evaluator import evaluate_single_image
import os
from script.visualize import DataVisualizer

def trans_img(imgs):
    imgs = imgs.transpose((2,0,1,3))
    saveimg = np.zeros((imgs.shape[0],imgs.shape[1],imgs.shape[2],3),dtype=float)
    for i in range(imgs.shape[0]):
        img = np.argmax(imgs[i], axis=2)
        saveimg[i][img == 0] = [0, 0, 0]
        saveimg[i][img == 1] = [0, 0, 128]
        saveimg[i][img == 2] = [0, 128, 0]
        saveimg[i][img == 3] = [0, 128, 128]
        saveimg[i][img == 4] = [128, 0, 0]
        saveimg[i][img == 5] = [128, 0, 128]
    return saveimg.transpose((1,2,0,3))


def evaluate_results( gt, y_pred, patient_ids, x):
    evaluation_list = []
    auc_roc_list = []
    auc_pr_list = []
    dice_coeff_list = []
    acc_list = []
    sensitivity_list = []
    specificity_list = []
    binary_image, auc_roc, auc_pr, dice_coeff, acc, sensitivity, specificity = \
        evaluate_single_image(y_pred[0], gt[0])
    auc_roc_list.append(auc_roc)
    auc_pr_list.append(auc_pr)
    dice_coeff_list.append(dice_coeff)
    acc_list.append(acc)
    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)
    pred_bin = binary_image.transpose((3,0,1,2))
    gt_trans = np.squeeze(gt).transpose((3,0,1,2))
    for i in range(gt_trans.shape[0]):
        pred_area = np.expand_dims(pred_bin[i],axis=3)
        gt_area = np.expand_dims(gt_trans[i],axis=3)
        _, auc_roc, auc_pr, dice_coeff, acc, sensitivity, specificity = \
            evaluate_single_image(pred_area, gt_area)
        auc_roc_list.append(auc_roc)
        auc_pr_list.append(auc_pr)
        dice_coeff_list.append(dice_coeff)
        acc_list.append(acc)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
    result = {'patient_id': patient_ids[0],
              'bin': binary_image,
              'x': x[0],
              'gt': gt[0],
              'acc': acc_list,
              'sn': sensitivity_list,
              'sp': specificity_list,
              'dice': dice_coeff_list,
              'auc_roc': auc_roc_list,
              'auc_pr': auc_pr_list}
    evaluation_list.append(result)
    return evaluation_list

def save_result(evaluated_results,save_path,epoch):
    if not os.path.exists(save_path+"/%04d" % (epoch)):
        os.mkdir(save_path+"/%04d" % (epoch))

    target = open(save_path + "/%04d/val.csv" % (epoch), "w")

    target.write('patient,dice_background,dice_Super,dice_Later,dice_Media，dice_Active,dice_Iner,dice_Mean\n')

    for i, evaluated_result in enumerate(evaluated_results):
        Dice_mean = (evaluated_result['dice'][1]+evaluated_result['dice'][2]+evaluated_result['dice'][3]
                     +evaluated_result['dice'][4]+evaluated_result['dice'][5]+evaluated_result['dice'][6])/6
        target.write("%s,%f,%f,%f,%f,%f,%f,%f\n" % (evaluated_result['patient_id'],
                                                 evaluated_result['dice'][1],
                                                 evaluated_result['dice'][2],
                                                 evaluated_result['dice'][3],
                                                 evaluated_result['dice'][4],
                                                 evaluated_result['dice'][5],
                                                 evaluated_result['dice'][6],
                                                 Dice_mean))

        np.save(file=save_path + "/%04d/" % (epoch) +"{}_x.npy".format(evaluated_result['patient_id']),
                arr=evaluated_result['x'])
        np.save(file=save_path + "/%04d/" % (epoch) +"{}_gt.npy".format(evaluated_result['patient_id']),
                arr=evaluated_result['gt'])
        np.save(file=save_path + "/%04d/" % (epoch) +"{}_bin.npy".format(evaluated_result['patient_id']),
                arr=evaluated_result['bin'])
        dv = DataVisualizer([np.squeeze(evaluated_result['x']),
                             trans_img(np.squeeze(evaluated_result['bin'])),
                             trans_img(np.squeeze(evaluated_result['bin'])),
                             trans_img(np.squeeze(evaluated_result['gt']))],
                            save_path= save_path+ "/%04d/" % (epoch) +"{}.png".format( evaluated_result['patient_id']))
        dv.visualize(evaluated_result['x'].shape[2])
    target.close()
