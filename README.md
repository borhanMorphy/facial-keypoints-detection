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

### Pre-training

### Training

### Test And Submission

### Deployment