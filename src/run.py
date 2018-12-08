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
    net_name_list_2048 = ['ResNet18', 'ResNet34', 'VGG16', 'InceptionV3']
    net_name_list_1024 = ['ResNet50', 'ResNet101', 'VGG19', 'Xception', 'DenseNet121', 'DenseNet169',
                          'InceptionResNetV2', 'NASNetMobile']
    net_name_list_512 = ['DenseNet201', 'ResNet152']

    for net_name in net_name_list_512:
        training_predict_path = get_training_predict_path(net_name)
        test_predict_path = get_test_predict_path(net_name)

        if not os.path.exists(training_predict_path):
            os.mkdir(training_predict_path)

        if not os.path.exists(test_predict_path):
            os.mkdir(test_predict_path)

        job_name = net_name
        command = "bsub -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
                  "./cluster_err/" + job_name + "_err.txt "
        command += "-W 24:00 -M 102400 -S 100 -P nova0019 -gpu - -R gpu ./predict.zsh "
        os.system(command + " " + net_name)


def run_training():
    net_name_list = ['ResNet50', 'ResNet101',
                     'VGG16', 'VGG19',
                     'Xception', 'InceptionResNetV2', 'NASNetMobile',
                     'InceptionV3']
    net_name_list = ['DenseNet121', 'DenseNet169', 'DenseNet201']
    net_name_list = ['GapNet-PL']
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
            command += "-W 120:00 -M 60000 -S 100 -P nova0019 -gpu \"num=2\" -R gpu ./train.zsh "
            os.system(command + " " + net_name + " " + str(k_fold))


def run_evaluate():
    net_name_list = ['DenseNet201']

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
