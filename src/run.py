import os
import argparse

from utils import get_acc_loss_path, get_history_path, get_weights_path
from utils import generate_exp_config
from utils import get_submission_path
from utils import get_training_predict_path
from utils import get_test_predict_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--training', default=False, action='store_true')
    parser.add_argument('-p', '--predict', default=False, action='store_true')
    parser.add_argument('-e', '--evaluate', default=False, action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.training:
        run_training()

    if args.predict:
        run_predict()

    if args.evaluate:
        run_evaluate()


def run_predict():
    net_name_list = ['ResNet18']

    for net_name in net_name_list:
        training_predict_path = get_training_predict_path(net_name)
        test_predict_path = get_test_predict_path(net_name)

        if not os.path.exists(training_predict_path):
            os.mkdir(training_predict_path)

        if not os.path.exists(test_predict_path):
            os.mkdir(test_predict_path)

        job_name = net_name
        command = "bsub -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
                  "./cluster_err/" + job_name + "_err.txt "
        command += "-W 120:00 -M 60000 -S 100 -P nova0019 -gpu - -R gpu ./predict.zsh "
        os.system(command + " " + net_name)


def run_training():
    net_name_list = ['DenseNet169', 'DenseNet201', 'InceptionResNetV2', 'InceptionV3', 'Xception', 'VGG16', 'VGG19']
    kfold_list = [0, 1, 2, 3, 4]

    for net_name in net_name_list:
        history_path = get_history_path(net_name=net_name)
        weights_path = get_weights_path(net_name=net_name)
        acc_loss_path = get_acc_loss_path(net_name=net_name)

        # create the output path
        if not os.path.exists(history_path):
            os.mkdir(history_path)

        if not os.path.exists(weights_path):
            os.mkdir(weights_path)

        if not os.path.exists(acc_loss_path):
            os.mkdir(acc_loss_path)

        # run
        for k_fold in kfold_list:
            exp_config = generate_exp_config(net_name, k_fold)
            job_name = exp_config
            command = "bsub -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
                      "./cluster_err/" + job_name + "_err.txt "
            command += "-W 120:00 -M 80000 -S 100 -P nova0019 -gpu \"num=2\" -R gpu ./train.zsh "
            os.system(command + " " + net_name + " " + str(k_fold))


def run_evaluate():
    net_name_list = ['ResNet18']

    for net_name in net_name_list:
        submission_path = get_submission_path(net_name=net_name)
        # create the output path
        if not os.path.exists(submission_path):
            os.mkdir(submission_path)

        job_name = net_name
        # command = "bsub -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
        #         "./cluster_err/" + job_name + "_err.txt "
        # command += "-W 8:00 -M 10240 -S 100 -P izkf ./evaluate.zsh "
        command = './evaluate.zsh '
        os.system(command + " " + net_name)


if __name__ == '__main__':
    main()
