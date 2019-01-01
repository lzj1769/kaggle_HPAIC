from __future__ import print_function

import sys
import requests
import pandas as pd

from configure import *


def main():
    colors = ['red', 'green', 'blue', 'yellow']
    v18_url = 'http://v18.proteinatlas.org/images/'

    img_list = pd.read_csv(HPAV18_CSV)
    print(len(img_list), file=sys.stderr)

    for i in img_list['Id']:
        img = i.split('_')
        for color in colors:
            img_path = img[0] + '/' + "_".join(img[1:]) + "_" + color + ".jpg"
            img_name = i + "_" + color + ".jpg"
            img_url = v18_url + img_path
            r = requests.get(img_url, allow_redirects=True)
            open(HPAV18_DIR + img_name, 'wb').write(r.content)


if __name__ == '__main__':
    main()
