# Sentiment Analysis of Movie Reviews

## Team Members

|First Name|Last Name|Index Number|
|----|--------|------|
|Dawid|Nowakowski|151868|
|Hubert|Sulżycki|151850|

## Prerequisites
- Python 3.10

## Repository Setup

1. **Clone the repository**
```bash
git clone https://github.com/dawidnowakowski/edan_project
cd edan_project/
```
2. **Create a virtual environment**
```bash
py -3.10 -m venv .venv
```
3. **Activate the virtual environment**
```bash
.venv\Scripts\activate.bat
```
4. **Install required libraries**
```bash
pip install -r requirements.txt
```
5. **Deactivate the virtual environment when done**
```bash
deactivate
```
<!--**To save current dependencies to the requirements file**
```bash
pip freeze > requirements.txt
```-->

## SVM

1. **Train the model**
```bash
cd src
python svm.py --task [training] --file [path to dataset csv file] --model [path to save model]
```
**The dataset must be a CSV file with two columns: `review` and `sentiment`.**

You can download a sample dataset [here](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

2. **Use a pre-trained model**
```bash
cd src
python svm.py --task [test] --file [path to review text file] --model [path to SVM model file]
```

You can download a sample pre-trained model [here](https://drive.google.com/drive/folders/1NwIchx53lbwrQFVLx_6dsAhWjmD_oKwC?usp=sharing).

## BERT

1. **Train the model**
```bash
cd src
python bert.py --task [training] --file [path to dataset csv file] --model [path to save model]
```
**The dataset must be a CSV file with two columns: `review` and `sentiment`.**

You can download a sample dataset [here](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

2. **Use a pre-trained model**
```bash
cd src
python bert.py --task [test] --file [path to review textfile] --model [path to bert model folder]
```

You can download a sample pre-trained model [here](https://drive.google.com/drive/folders/1NwIchx53lbwrQFVLx_6dsAhWjmD_oKwC?usp=sharing).
