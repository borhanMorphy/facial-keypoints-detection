# Kaggle Facial Keypoints Detection Challenge
Facial keypoints detection with end-to-end development cycle

refered competition: [facial-keypoints-detection](https://www.kaggle.com/c/facial-keypoints-detection)

## Contents
  - [Introduction](#introduction)
  - [Project Setup](#project-setup)
  - [Data Discovery](#data-discovery)
  - [Pre-training](#pre-training)
  - [Training](#training)
  - [Test And Submission](#test-and-submission)
  - [Deployment](#deployment)

### Introduction
The aim of this repository is to explain how can someone take action from only data and push it to the production
<br>

### Project Setup
lets start with installing the requirements
```
pip3 install -r requirements.txt
```
now lets get the data from kaggle.<br>
(PS: assuming you already have a kaggle api, otherwise checkout [here])

[here]: https://github.com/Kaggle/kaggle-api
```
# create `data` folder
mkdir ./data

# download data into `./data` folder
kaggle competitions download -c facial-keypoints-detection -p data

# unzipping
unzip ./data/facial-keypoints-detection -d ./data/
unzip ./data/training.zip -d ./data/
unzip ./data/test.zip -d ./data/
rm ./data/*.zip
```
### Data Discovery
In the competition author explicitly says that, some of the data is missing
> In some examples, some of the target keypoint positions are misssing (encoded as missing entries in the csv, i.e., with nothing between two commas)

Run the following script to see which of the fields have missing data, how many row is missing and do missing rows overlaps with each other or not
```
python3 check_missings.py -i ./data/training.csv
```
When we look at the output;<br>
with given 30 features<br>
2 features does not contain missing data<br>
6 features does contain missing data but missing data percentage is below %1<br>
other 22 features does contain missing data and missing data percentage is between %65 - %70<br>

### Pre-training
In this section we will handle missing data before actual training and extract images/annotations from csv file.<br>
to fix missing data, we will follow 2 approaches.<br>
1) for features that had missing data percentage is ~ %1, we will use mean values.
2) for features that had missing data percentage is ~ %67, we will use linear regression.<br>
with help of non-missing features, we will predict missing values

to achive these steps run the following script, which will fix the missing values and save the new csv file as `training_fixed.csv`
```
# run with --debug flag if you want to see the prediction result
python3 fix_missings.py -i ./data/training.csv --debug
```

to extract images and annotations from csv file use the following script
```
# this script will extract data to ./data/training_fixed/ and ./data/test/ folder
python3 csv2images.py --train-input ./data/training_fixed.csv --test-input ./data/test.csv
```

### Training

### Test And Submission

### Deployment