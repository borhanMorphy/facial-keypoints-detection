import pandas as pd
import numpy as np
import argparse

def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input','-i',type=str,default='./data/training.csv',help='give training csv file path')

    return ap.parse_args()

def check(key:str, arr1:np.ndarray, arr2:np.ndarray):
    print(f"processing {key}")
    miss1, = np.where(np.isnan(arr1))
    miss2, = np.where(np.isnan(arr2))

    total_arr1 = arr1.shape[0]
    total_miss1 = miss1.shape[0]

    total_arr2 = arr2.shape[0]
    total_miss2 = miss2.shape[0]

    print(f"total missing values for key {key}_x is: {total_miss1}\t \
        missing percent: {((total_miss1*100)/total_arr1):.02f}")
    
    print(f"total missing values for key {key}_y is: {total_miss2}\t \
        missing percent: {((total_miss2*100)/total_arr2):.02f}")
    
    print(f"percent of t")
    print(f"missing values matches: {(miss1 == miss2).all()}\n")

def main(train_csv_path:str):
    train_df = pd.read_csv(train_csv_path)
    del train_df['Image']

    labels = [
        'left_eye_center',
        'right_eye_center',
        'left_eye_inner_corner',
        'left_eye_outer_corner',
        'right_eye_inner_corner',
        'right_eye_outer_corner',
        'left_eyebrow_inner_end',
        'left_eyebrow_outer_end',
        'right_eyebrow_inner_end',
        'right_eyebrow_outer_end',
        'nose_tip',
        'mouth_left_corner',
        'mouth_right_corner',
        'mouth_center_top_lip',
        'mouth_center_bottom_lip']
    
    for label in labels:
        arr1 = train_df[label+"_x"].to_numpy()
        arr2 = train_df[label+"_y"].to_numpy()
        check(label,arr1,arr2)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input)