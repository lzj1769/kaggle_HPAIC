import os
import argparse

from utils import get_acc_loss_path, get_logs_path, get_weights_path
from utils import generate_exp_config
from utils import get_training_predict_path, get_test_predict_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default=False, action='store_true')
    parser.add_argument("--predict", default=False, action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.training:
        run_training()

    if args.predict:
        run_predict()


def run_predict():
    net_name_list = ['ResNet50']
    pre_trained_list = [0, 1]
    optimizer_list = [0, 1, 2]

    kfold_list = [0, 1, 2, 3, 4, 5, 6, 7]

    for net_name in net_name_list:
        training_predict_path = get_training_predict_path(net_name=net_name)
        test_predict_path = get_test_predict_path(net_name=net_name)

        # create the output path
        if not os.path.exists(training_predict_path):
            os.mkdir(training_predict_path)

        if not os.path.exists(test_predict_path):
            os.mkdir(test_predict_path)
        # run
        for pre_trained in pre_trained_list:
            for optimizer in optimizer_list:
                for k_fold in kfold_list:
                    exp_config = generate_exp_config(net_name, pre_trained, optimizer, k_fold)
                    job_name = exp_config
                    # command = "bsub -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
                    #          "./cluster_err/" + job_name + "_err.txt "
                    # command += "-W 8:00 -M 102400 -S 100 -P nova0019 -gpu - -R gpu ./predict.zsh "
                    command = "./predict.zsh "
                    os.system(
                        command + " " + net_name + " " + str(pre_trained) + " " + str(optimizer) + " " + str(k_fold))


def run_training():
    net_name_list = ['ResNet50']
    pre_trained_list = [0]
    optimizer_list = [0, 1, 2, 3, 4, 5, 6, 7]

    kfold_list = [0]

    for net_name in net_name_list:
        logs_path = get_logs_path(net_name=net_name)
        weights_path = get_weights_path(net_name=net_name)
        acc_loss_path = get_acc_loss_path(net_name=net_name)

        # create the output path
        if not os.path.exists(logs_path):
            os.mkdir(logs_path)

        if not os.path.exists(weights_path):
            os.mkdir(weights_path)

        if not os.path.exists(acc_loss_path):
            os.mkdir(acc_loss_path)

        # run
        for pre_trained in pre_trained_list:
            for optimizer in optimizer_list:
                for k_fold in kfold_list:
                    exp_config = generate_exp_config(net_name, pre_trained, optimizer, k_fold)
                    job_name = exp_config
                    command = "bsub -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
                              "./cluster_err/" + job_name + "_err.txt "
                    command += "-W 8:00 -M 102400 -S 100 -P nova0019 -gpu - -R gpu ./train.zsh "
                    os.system(
                        command + " " + net_name + " " + str(pre_trained) + " " + str(optimizer) + " " + str(k_fold))


if __name__ == '__main__':
    main()
