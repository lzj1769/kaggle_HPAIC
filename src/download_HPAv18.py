from __future__ import print_function

import pandas as pd
import os
import sys
from urllib import urlopen
from lxml import etree
import requests
from configure import *


def fill_targets(row):
    labels = []
    labels_all = []

    reverse_train_labels = dict((v, k) for k, v in NAME_LABEL_DICT.items())

    for name in row["GO id"].split(";"):
        row.loc[name] = 1
        label = reverse_train_labels[name]
        labels_all.append(label)
        if label < 28:
            labels.append(label)

    row.loc['Target_all'] = ' '.join(str(e) for e in labels_all)
    row.loc['Target'] = ' '.join(str(e) for e in labels)
    return row


def main():
    GeneList = pd.read_csv(SUBCELLULAR_LOCATION_CSV, sep='\t')
    GeneList['Target_all'] = ""
    GeneList['Target'] = ""

    for label_names in NAME_LABEL_DICT:
        GeneList[NAME_LABEL_DICT[label_names]] = 0

    GeneList = GeneList.apply(fill_targets, axis=1)

    HPAv18_metadata = GeneList[GeneList['Target'] != ""]

    # download the images
    HPAv18_data = []

    colors = ['red', 'green', 'blue', 'yellow']

    def download_image(gene, target):
        xml = urlopen('https://www.proteinatlas.org/%s.xml' % gene)
        tree = etree.parse(xml)
        imageUrls = tree.xpath('//cellExpression/subAssay/data/assayImage/image/imageUrl/text()')

        for imageUrl in imageUrls:
            path = imageUrl.split("/")
            Identitiy = gene + '_' + path[4] + '_' + path[5].replace("_blue_red_green.jpg", '')
            HPAv18_data.append({"Id": Identitiy, "Target": target})
            for c in colors:
                img_url = imageUrl.replace("blue_red_green", c)
                path = img_url.split("/")
                save_name = gene + '_' + path[4] + '_' + path[5]
                r = requests.get(img_url, allow_redirects=True)
                filename = os.path.join(HPAV18_DIR, save_name)
                open(filename, 'wb').write(r.content)

    for index, row in HPAv18_metadata.iterrows():
        download_image(row['Gene'], row['Target'])
        print(index, row['Gene'], file=sys.stderr)

    df = pd.DataFrame(HPAv18_data)
    df.to_csv(HPAV18_CSV, index=False)


if __name__ == '__main__':
    main()
