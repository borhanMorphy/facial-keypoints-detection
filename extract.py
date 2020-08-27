import pandas as pd
import os
from cv2 import cv2
import numpy as np
from typing import Tuple
import json

def str2img(str_img:str, img_size:Tuple=(96,96)):
    return np.array([int(pixel) for pixel in str_img.split(" ")]).astype(np.uint8).reshape(img_size)

def main(file_name:str):
    fname = os.path.splitext(file_name)[0]
    if not os.path.exists(fname):
        os.mkdir(fname)
    df = pd.read_csv(file_name)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    for row_id,row in df.iterrows():
        base_file_name = os.path.join(fname,str(row_id))
        row = json.loads(row.to_json())
        img = row.pop('Image')
        img = str2img(img)
        labels = []
        for k,v in row.items():
            labels.append(f"{k} {v}")
        with open(base_file_name+".txt","w") as foo:
            foo.write("\n".join(labels))
        cv2.imwrite(base_file_name+".jpg",img)

if __name__ == "__main__":
    import sys
    main(sys.argv[1])