from setuptools import setup, find_packages

setup(
    name="svm_lib",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "pandas",
        "nltk",
        "matplotlib",
        "joblib",
        "tqdm"
    ],
)
