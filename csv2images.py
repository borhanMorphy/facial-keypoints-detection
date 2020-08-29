import pandas as pd
import os
from cv2 import cv2
import numpy as np
import json
import argparse
from tqdm import tqdm
from utils import str2img

def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-input',type=str,
        default='./data/training_fixed.csv', help='give training csv file path')

    ap.add_argument('--test-input', type=str,
        default='./data/test.csv', help='give test csv file path')
    return ap.parse_args()

def main(*csv_files):
    for csv_file in csv_files:
        print(f"extracting {csv_file}")
        extract(csv_file)

def extract(file_name:str):
    fname = os.path.splitext(file_name)[0]
    if not os.path.exists(fname):
        os.mkdir(fname)
    df = pd.read_csv(file_name)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    for row_id,row in tqdm(df.iterrows(), total=len(df)):
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
    args = parse_arguments()
    main(args.train_input,args.test_input)