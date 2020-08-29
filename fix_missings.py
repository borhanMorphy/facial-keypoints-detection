import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import List,Tuple
from cv2 import cv2
import os
from utils import str2img

def complete_minor_missings(minor_missings:List, df):
    minors = []
    for minor_missing in minor_missings:
        minor_x = df[minor_missing+"_x"].to_numpy()
        minor_y = df[minor_missing+"_y"].to_numpy()

        miss_mask_x = np.isnan(minor_x)
        miss_mask_y = np.isnan(minor_y)

        # complete missing values with mean
        minor_x[miss_mask_x] = minor_x[~miss_mask_x].mean()
        minor_y[miss_mask_y] = minor_y[~miss_mask_y].mean()

        minor = np.stack([minor_x,minor_y],axis=1)
        minors.append(minor)

    # N,6
    return np.concatenate(minors,axis=1)

def complete_major_missings(major_missings:List, features:np.ndarray, df, images):
    # split as train / test using missing values
    # train linear regression model for prediction
    # predict missing values
    # set missing values
    # next
    majors = []
    for major_missing in major_missings:
        print(f"training for {major_missing}")
        major_x = df[major_missing+"_x"].to_numpy()
        major_y = df[major_missing+"_y"].to_numpy()

        miss_mask_x = np.isnan(major_x)
        miss_mask_y = np.isnan(major_y)

        # N,8 => M,8
        X_train = features[~miss_mask_x, :]
        # N,8 => (N-M),8
        X_test = features[miss_mask_x, :]
        # M,
        y_train = major_x[~miss_mask_x]

        model = LinearRegression().fit(X_train, y_train)

        major_x[miss_mask_x] = model.predict(X_test)
        ####################### train y #####################3

        # N,8 => M,8
        X_train = features[~miss_mask_y, :]
        # N,8 => (N-M),8
        X_test = features[miss_mask_y, :]
        # M,
        y_train = major_y[~miss_mask_y]

        model = LinearRegression().fit(X_train, y_train)

        major_y[miss_mask_y] = model.predict(X_test)

        major = np.stack([major_x,major_y],axis=1) # N,2
        majors.append(major)
        
        """
        # for debug
        indexes, = np.where(miss_mask_x)
        for img,landmark in zip(images.iloc[indexes],major[miss_mask_x,:]):
            img = str2img(img)
            res = draw_img(img,landmark*96)
            if res == 27:
                exit(0)
        """
    return np.concatenate(majors,axis=1)

def draw_img(img,landmark):
    bgr = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    x,y = landmark.astype(np.int32)
    bgr = cv2.circle(bgr,(x,y),2,(0,0,255))
    cv2.imshow("",bgr)
    return cv2.waitKey(0)

def main(training_csv_file:str):
    minor_missings = ['left_eye_center','right_eye_center','mouth_center_bottom_lip']
    major_missings = [
        'left_eye_inner_corner','left_eye_outer_corner',
        'right_eye_inner_corner','right_eye_outer_corner',
        'left_eyebrow_inner_end','left_eyebrow_outer_end',
        'right_eyebrow_inner_end','right_eyebrow_outer_end',
        'mouth_left_corner','mouth_right_corner','mouth_center_top_lip']
    non_missing = 'nose_tip'

    df = pd.read_csv(training_csv_file)
    images = df['Image']
    del df['Image']

    df /= 96 # fit between [0,1]
    
    # N,2
    source_keypoints = np.stack([df[non_missing+'_x'].to_numpy(), df[non_missing+'_y'].to_numpy()], axis=1)

    # N,6
    minors = complete_minor_missings(minor_missings,df)

    # N,8
    features = np.concatenate([source_keypoints,minors],axis=1)

    # N,22
    majors = complete_major_missings(major_missings,features,df,images)

    # N,30
    features = np.concatenate([features,majors],axis=1)

    headers = []
    for header in [non_missing] + minor_missings + major_missings:
        headers.append(header+"_x")
        headers.append(header+"_y")
    headers.append('Image')

    features = (features * 96).astype(np.float32)
    features = [features[:,i] for i in range(features.shape[1])]
    features.append(images)
    altered_df = pd.DataFrame(zip(*features), columns=headers)

    new_file_path = os.path.join(os.path.dirname(training_csv_file), 'training_fixed.csv')
    altered_df.to_csv(new_file_path)

if __name__ == "__main__":
    import sys
    main(sys.argv[1])