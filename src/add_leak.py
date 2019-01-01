from __future__ import print_function

import sys

import pandas as pd

from configure import *

input_file = sys.argv[1]
output_file = sys.argv[2]

leak_df = pd.read_csv(LEAK_FILE)
label_df = pd.read_csv(HPAV18_CSV)
input_df = pd.read_csv(input_file)


for i in range(leak_df.shape[0]):
    for j in range(input_df.shape[0]):
        if input_df['Id'][j] == leak_df['Test'][i]:
            test_id = leak_df['Test'][i]
            extra_id = leak_df['Extra'][i].split("_")
            extra_id2 = "_".join(extra_id[1:])
            label1 = input_df['Predicted'][j]
            idx = label_df[label_df['Id'] == extra_id2]['Target'].tolist()
            input_df['Predicted'][j] = idx[0]
            label2 = input_df['Predicted'][j]

            print("Original: {}, Leak {}".format(label1, label2))

input_df.to_csv(output_file, index=None)