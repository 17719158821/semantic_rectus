import numpy as np
import os
from utils import evaluate_results



data_pred_path = "./data_pred/"
data_y_path = "./data_y/"
data_x_path = "./data_x/"
save_path = "./save/"

def dice_evaluate(data_pred,data_y,data_x,patientids):
    result = evaluate_results(data_y,data_pred,patientids,data_x)
    dice_mean =  (result[0]["dice"][1]+result[0]["dice"][2]+result[0]["dice"][3]
                     +result[0]["dice"][4]+result[0]["dice"][5]+result[0]["dice"][6])/6
    return result,dice_mean

