import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3
import nltk
import re
nltk.download(['punkt','stopwords','wordnet'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
import pickle


def load_data(database_filepath):
    """
    INPUT:
        database_filepath (string) : database location
    OUTPUT:
        X  : messages to process
        Y  : training/evaluating categories
        labels  : list of message classification labels
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM Messages", engine)
    
    X = df['message'] 
    Y = df.drop(['message', 'genre', 'id', 'original'], axis = 1)
    return X, Y,list(Y.columns)

def tokenize(text):
    """
    Remove capitalization and special characters and lemmatize texts
    Args:
    text: string. String containing message for processing
    
    Returns:
    clean_tokens: list of strings. 
    """ 
    # Convert text to lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    Function: build model that consist of pipeline
    Args:
      N/A
    Return
      cv(model): Grid Search model 
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 10,
                                                             min_samples_split = 10)))
    ])
    
    parameters = {#'vect__min_df': [1, 5],
              'tfidf__use_idf':[True, False],
              'clf__estimator__n_estimators':[10, 50], 
              'clf__estimator__min_samples_split':[2, 5, 10]}

    cv = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1,verbose=2)

   # cv.fit(X_train, y_train)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
   y_pred = model.predict(X_test)
   print(classification_report(Y_test.iloc[:,1:].values, np.array([x[1:] for x in y_pred]),     target_names=category_names))


def save_model(model, model_filepath):
    """
    Function: save model as pickle file.
    Args:
      cv:target model
    Return:
      N/A
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)   


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()