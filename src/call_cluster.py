import os

net_list = ['SimpleNet_LogLoss',
            'PreTrained_DenseNet_121_LogLoss',
            'PreTrained_DenseNet_169_LogLoss',
            'PreTrained_DenseNet_201_LogLoss',
            'PreTrained_InceptionResNetV2_LogLoss',
            'PreTrained_InceptionV3_LogLoss',
            'PreTrained_MobileNet_LogLoss',
            'PreTrained_MobileNetV2_LogLoss',
            'PreTrained_NASNetLarge_LogLoss',
            'PreTrained_NASNetMobile_LogLoss',
            'PreTrained_ResNet_50_LogLoss',
            'PreTrained_Xception_LogLoss',
            'TrainedFromScratch_DenseNet_121_LogLoss',
            'TrainedFromScratch_DenseNet_169_LogLoss',
            'TrainedFromScratch_DenseNet_201_LogLoss',
            'TrainedFromScratch_InceptionResNetV2_LogLoss',
            'TrainedFromScratch_InceptionV3_LogLoss',
            'TrainedFromScratch_MobileNet_LogLoss',
            'TrainedFromScratch_MobileNetV2_LogLoss',
            'TrainedFromScratch_NASNetLarge_LogLoss',
            'TrainedFromScratch_NASNetMobile_LogLoss',
            'TrainedFromScratch_ResNet_50_LogLoss',
            'TrainedFromScratch_Xception_LogLoss']

net_list = ['PreTrained_MobileNet_LogLoss']
for net in net_list:
    job_name = net
    command = "bsub -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
              "./cluster_err/" + job_name + "_err.txt "
    command += "-W 8:00 -M 102400 -S 100 -R \"select[hpcwork]\" -P nova0019 -gpu - -R gpu ./train.zsh "
    os.system(command + " " + net)
