#-----------------------------------Importing the necessary dependencies

import pandas as pd
import spacy
import gensim
from gensim.models import Word2Vec
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer #importing Tfidf from sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

#-----------------------------------Importing the Dataset

dataset_url = "https://raw.githubusercontent.com/osamuzahid/NLP/refs/heads/main/bbc-text.csv" #loading the dataset, which was uploaded to github
df = pd.read_csv(dataset_url) #reading the data from the .csv file in the url and loading it into a dataframe for further analysis and manipulation

print (df.head(10)) #printing the first few rows to ensure the dataframe was loaded properly

#----------------------------------Preprocessing Data With Spacy

nlp = spacy.load("en_core_web_sm") #loading spacys English pipeline. We'll use web_sm, which is leaner and optimized for efficiency, instead of web_trf which is larger and slower but more accurate

def text_preprocessing(text): #processing text by removing stop words, removing punctuation, and lemmatizing each word
  doc = nlp(text)
  processed_text = ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
  return processed_text

df['processed_text'] = df['text'].apply(text_preprocessing) #applying our preprocessing function to the 'text' column of the dataframe and creating a new column 'processed_text' containing the preprocessed text

def token_split(text): #a function to take an input string and return a list of individual words (tokens)
  return list(text.split())
df.columns
df['processed_and_tokenized_text'] = df['processed_text'].apply(token_split) #applying the token_split function to the processed text to get individual tokens from the previously preprocessed strings

#----------------------------------Using NER as the second of our first of three features

def named_entity_counts(row): #creating a function to count the relevant named entities for each category
    text = row['processed_text']
    category = row['category']

    # Defining the relevant entities for each category
    if category == 'politics':
        relevant_entities = ["PERSON", "ORG", "GPE", "EVENT"]
    elif category == 'business':
        relevant_entities = ["ORG", "PRODUCT", "MONEY", "GPE"]
    elif category == 'sport':
        relevant_entities = ["PERSON", "ORG", "EVENT", "MONEY"]
    elif category == 'entertainment':
        relevant_entities = ["PERSON", "WORK_OF_ART", "ORG", "EVENT"]
    elif category == 'tech':
        relevant_entities = ["ORG", "PRODUCT", "GPE", "MONEY"]
    else:
        relevant_entities = []


    entity_counts = {entity: 0 for entity in relevant_entities} # Initializing the entity counts dictionary

    # Processing the text and count relevant entities
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in relevant_entities:
            entity_counts[ent.label_] += 1

    return entity_counts


df['entity_counts'] = df.apply(named_entity_counts, axis=1) # Applying the function to each row


#----------------------------------Using TFIDF as the second of our three features



tfidf = TfidfVectorizer(max_features = 1000) #Creating an instance of the Tfidf vectorizer and limiting it to a maximum of 1000 features

tfidf_vectors = tfidf.fit_transform(df['processed_text']) #Applying TFIDF transformaton to the data in the 'processed__text' column, converting it into a sparse matrix of numerical features.

df_tfidf = pd.DataFrame(tfidf_vectors.toarray()) # Converting the sparse matrix into a dense array and turning it into a dataframe

df = pd.concat([df, df_tfidf], axis=1) #Combining the original dataframe with the new tf_idf dataframe, adding tfidf features as new columns to the original dataframe

print (df.head(10)) #Printing a few lines to ensure everything is working

#----------------------------------Using Word Embeddings (implemented via Word2Vec) as the second of our three features

# Training the Word2Vec model
tokenized_docs = df['processed_and_tokenized_text'].tolist()
model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)

#Creating a function to get Word2Vec embeddings
def get_word2vec_embeddings(tokens, model):
    word_vectors = []
    for token in tokens:
        if token in model.wv:  # Ensuring the token exists in the Word2Vec model's vocabulary
            word_vectors.append(model.wv[token])  # Appending the word vector to the list
    if word_vectors:  # If word vectors exist, returning their average
        return np.mean(word_vectors, axis=0)
    else:  # If no word vectors found, returning a zero vector
        return np.zeros(model.vector_size)


def apply_word2vec(row): # Applying the previous function to extract Word2Vec embeddings
    return get_word2vec_embeddings(row, model)

df['word2vec_embeddings'] = df['processed_and_tokenized_text'].apply(apply_word2vec) #Using the function on the processed and tokenized text column to get word2vec embeddings from our data

#----------------------------------Concatenating all our features into one dataframe

df_entity = pd.json_normalize(df['entity_counts']) # Creating a dataframe containing the 'entity_counts' column converted to individual columns for each entity type

df_word2vec = pd.DataFrame(df['word2vec_embeddings'].to_list()) #Creating a dataframe with all word2vec embeddings

#We previously created a dataframe in step 2.2 containing our tfidf vectors named df_tfidf

df_all_features = pd.concat([df_entity, df_tfidf, df_word2vec], axis = 1) #creating a new dataframe with all the features

df_all_features = pd.concat([df['category'], df_all_features], axis = 1) #adding the category column to our new dataframe with all the features

df_all_features = df_all_features.fillna(0) #filling up the N/A spaces in our data frame with zeroes

print (df_all_features.head(10)) #printing a few rows to ensure everything is working as expected


#-------------------------------Scaling the features to ensure they contribute equally to model performance.

X = df_all_features.drop('category', axis=1)  # Removing the target column to separate it from features
X.columns = X.columns.astype(str)
scaler = StandardScaler()  # Create a StandardScaler object
X_scaled = scaler.fit_transform(X)  # Apply scaling to the features

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

#-------------------------------Performing feature selection using RFE to get the top features from our chosen features

X = X_scaled_df  # Using the dataframe containing scaled features
y = df_all_features['category']  # Setting y as the target category column

encoder = LabelEncoder()  # Creating a LabelEncoder object
y_encoded = encoder.fit_transform(y)  # Encoding the target categories as numerical values

X.columns = X.columns.astype(str) #converting all columns in X to strings to avoid type errors during feature selection

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42) #splitting the data into training and test sets with an 80/20 distribution

model = RandomForestClassifier(n_estimators=100, random_state=42) #creating the RandomForestClassifier model object with 100 decision trees

selector = RFE(estimator=model, n_features_to_select=100) #initializing RFE using the RandomForestClassifier model and choosing the top 100 features

selector.fit(X_train, y_train) #Fitting the training data to determine the top 100 features

selected_features = X_train.columns[selector.get_support()] #Acquiring the selected features

print(selected_features) #Printing the selected features to check that everything is working as expected

#-------------------------------Splitting the data and training our model


#Splitting the data into a 70% training set and 30% temp set, which we'll split 15%/15% into training and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled_df[selected_features], y_encoded, test_size=0.3, random_state=42)


X_test, X_dev, y_test, y_dev = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) #Splitting the temp set into development and test sets

#Training the model
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

#-------------------------------Evaluating the model and printing the results

y_dev_pred = model.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, y_dev_pred) #Evaluating on the development set
print(f"Development Set Accuracy: {dev_accuracy:.3f}")
print("Performance Report:\n", classification_report(y_dev, y_dev_pred))


y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred) #Evaluating on the development set
print(f"Test Set Accuracy: {test_accuracy:.3f}")
print("Performance Report:\n", classification_report(y_test, y_test_pred))

#----------------------------------------------------------------------------------------
