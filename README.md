News Article Classification

This project trains a machine learning model to categorize news articles into the following topics:

- politics
- business
- sports
- entertainment
- technology

**--------Prerequisites--------**

The following steps demonstrate how to run the model in a Linux environment.

Before running the NewsArticleClassification.py file, ensure your system has the following:


- Python (Preferably 3.8 or later)
  
- Internet Access
  
- PIP (To install necessary libraries/dependencies) (Install with 'sudo apt-install python3 pip' command)
  
- GIT (To clone this repository) (Install with 'sudo apt install git' command)

**--------Setup--------**

- Start a virtual environment using the following commands in the terminal:

python3 -m venv .venv

source .venv/bin/activate

- Install the necessary libararies by running the following command in the terminal:

pip install pandas spacy gensim scikit-learn numpy

- Run the following command to download spaCy's English model:

python3 -m spacy download en_core_web_sm

**--------Execution--------**

1. Clone the repository and navigate to its directory by running the following commands:

git clone https://github.com/osamuzahid/NLP-News-Classification.git

cd NLP-News-Classification

2. Run the Python file with the following command:

python3 NewsArticleClassification.py

   


