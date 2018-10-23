import os

net_list = ['SimpleNet_FocalLoss',
            'PreTrained_DenseNet_121_FocalLoss',
            'PreTrained_DenseNet_169_FocalLoss',
            'PreTrained_DenseNet_201_FocalLoss',
            'PreTrained_InceptionResNetV2_FocalLoss',
            'PreTrained_InceptionV3_FocalLoss',
            'PreTrained_MobileNet_FocalLoss',
            'PreTrained_MobileNetV2_FocalLoss',
            'PreTrained_NASNetLarge_FocalLoss',
            'PreTrained_NASNetMobile_FocalLoss',
            'PreTrained_ResNet_50_FocalLoss',
            'PreTrained_Xception_FocalLoss',
            'TrainedFromScratch_DenseNet_121_FocalLoss',
            'TrainedFromScratch_DenseNet_169_FocalLoss',
            'TrainedFromScratch_DenseNet_201_FocalLoss',
            'TrainedFromScratch_InceptionResNetV2_FocalLoss',
            'TrainedFromScratch_InceptionV3_FocalLoss',
            'TrainedFromScratch_MobileNet_FocalLoss',
            'TrainedFromScratch_MobileNetV2_FocalLoss',
            'TrainedFromScratch_NASNetLarge_FocalLoss',
            'TrainedFromScratch_NASNetMobile_FocalLoss',
            'TrainedFromScratch_ResNet_50_FocalLoss',
            'TrainedFromScratch_Xception_FocalLoss']

for net in net_list:
    job_name = net
    command = "bsub -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
              "./cluster_err/" + job_name + "_err.txt "
    command += "-W 8:00 -M 102400 -S 100 -R \"select[hpcwork]\" -P nova0019 -gpu - -R gpu ./train.zsh "
    os.system(command + " " + net)
