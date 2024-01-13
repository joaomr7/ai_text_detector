import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

import spacy
from spacy.tokens import Doc

from symspellpy import SymSpell, Verbosity
import importlib.resources

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.paths import ARTIFACTS_PATH

class TextPreprocessing(BaseEstimator, TransformerMixin):
    '''
    Custom sklearn transformer to apply text preprocessing in text data.
    '''
    def __init__(self):
        # setting up spacy and disabling unused pipeline components
        self.nlp = spacy.load('en_core_web_md', disable=['parser', 'ner', 'lemmatizer', 'textcat', 'custom'])

        # setting up SymSpell to typo count
        self.sym_spell = SymSpell(max_dictionary_edit_distance=1, prefix_length=7)

        # load symspell english dictionary
        with importlib.resources.open_text('symspellpy', 'frequency_dictionary_en_82_765.txt') as file:
            self.sym_spell.load_dictionary(file.name, term_index=0, count_index=1)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:

            # create spacy docs
            docs = [doc for doc in self.nlp.pipe(X)]

            text_vectors = []
            contractions_counts = []
            typos_counts = []

            contractions_patterns = [
                r'\b(\w+)\'(\w+)\b',
                r'\'(\w+)\b'
            ]

            for doc in docs:
                # count contractions
                contractions_count = 0
                for pattern in contractions_patterns:
                    matches = re.findall(pattern, doc.text)
                    contractions_count += len(matches)

                contractions_counts.append(contractions_count)

                # count typos
                typos_count = 0

                words = []
                spaces = []
                for token in doc:
                    if not re.match(r'[a-zA-Z]|\'[a-zA-Z]', token.text): # ignore irrelevant information
                        continue

                    if token.is_stop: # ignore stopwords
                        continue

                    # lookup for typo
                    correct_word = ''
                    suggestions = self.sym_spell.lookup(token.text, Verbosity.CLOSEST, max_edit_distance=1) # lookup for typo

                    if not suggestions:
                        correct_word = token.text
                        typos_count += 1

                    elif suggestions[0].term != token.text:
                        correct_word = suggestions[0].term
                        typos_count += 1

                    else:
                        correct_word = token.text

                    if token.pos_ in ['NOUN', 'PROPN']: # ignore nouns
                        continue

                    words.append(correct_word)
                    spaces.append(token.whitespace_)

                typos_counts.append(typos_count)

                clean_doc = Doc(vocab=self.nlp.vocab, words=words, spaces=spaces)
                text_vectors.append(clean_doc.vector)

            return np.column_stack([text_vectors, contractions_counts, typos_counts])
        
        except Exception as e:
            raise CustomException(e)

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(ARTIFACTS_PATH, 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        try:
            pipeline = Pipeline(
                steps=[
                    ('text_preprocessing', TextPreprocessing()),
                    ('scaler', MinMaxScaler())
                ]
            )

            return pipeline

        except Exception as e:
            raise CustomException(e)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            logging.info('Reading train and test data.')

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed.')

            logging.info('Obtaining preprocessing object')
            preprocessor_obj = self.get_data_transformer_object()

            input_feature_train_df = train_df.iloc[:, 0]
            target_feature_train_df = train_df.iloc[:, 1]

            input_feature_test_df = test_df.iloc[:, 0]
            target_feature_test_df = test_df.iloc[:, 1]

            logging.info('Applying preprocessing object to training and testing dataframes.')

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info('Saving preprocessing object.')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info('Saved preprocessing object.')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e)