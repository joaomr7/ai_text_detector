import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

import spacy
from spacy.tokens import Doc

from symspellpy import SymSpell, Verbosity
import importlib.resources

import textstat

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
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
            # tranfrom texts in spacy docs
            docs = [doc for doc in self.nlp.pipe(X)]

            text_vectors = []
            typos_counts = []
            reading_ease_scores = []
            lexical_diversity_scores = []

            def calculate_reading_ease_score(text):
                '''
                Helper function to calculate reading ease score.
                '''
                try:
                    return textstat.flesch_reading_ease(text)
                except ZeroDivisionError:
                    return 0.0
                
            def calculate_lexical_diversity_score(words_list):
                '''
                Helper function to calculate lexical diversity score from the words list.
                '''
                total_words_count = len(words_list)
                unique_words_count = len(set(words_list))

                if total_words_count == 0:
                    return 0.0
                else:
                    return unique_words_count / total_words_count

            for doc in docs:

                # reading ease score
                reading_ease_scores.append(calculate_reading_ease_score(doc.text))

                # count typos
                typos_count = 0

                nouns = []
                words = []
                spaces = []
                for token in doc:
                    if not token.is_alpha: # ignore irrelevant information
                        continue

                    if token.is_stop: # ignore stopwords
                        continue

                    if token.pos_ in ['NOUN', 'PROPN']: # ignore nouns
                        nouns.append(token.text.lower()) # consider nouns just to lexical diversity calculation
                        continue

                    # lookup for typo
                    word = token.text.lower()
                    suggestions = self.sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=1) # lookup for typo

                    # check if is a typo
                    if not suggestions or not any(suggestion.term == word for suggestion in suggestions):
                        typos_count += 1
                        continue

                    words.append(word)
                    spaces.append(token.whitespace_)

                # add typos count
                typos_counts.append(typos_count)

                # lexical diversity
                lexical_diversity_scores.append(calculate_lexical_diversity_score(words + nouns))

                # create and add text vector to list
                clean_doc = Doc(vocab=self.nlp.vocab, words=words, spaces=spaces)
                text_vectors.append(clean_doc.vector)

            # return numpy array
            return np.column_stack([text_vectors, typos_counts, reading_ease_scores, lexical_diversity_scores])

        except Exception as e:
            raise CustomException(e)

@dataclass
class DataTransformationConfig:
    '''
    Configuration class for storing file path to the preprocessor object.

    Attributes
    ---
    * preprocessor_obj_file_path: the file path to store the preprocessor object.
    '''

    preprocessor_obj_file_path: str = os.path.join(ARTIFACTS_PATH, 'preprocessor.pkl')

class DataTransformation:
    '''
    Class responsible for create the data preprocessor.
    '''

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        try:
            transformer = Pipeline(
                steps=[
                    ('text_preprocessing', TextPreprocessing()),
                    ('scaling', ColumnTransformer(
                        transformers=[
                            ('min_max_scaler', MinMaxScaler(), slice(0, 301)),
                            ('std_scaler', StandardScaler(), slice(301, 303))
                        ]
                    ))
                ]
            )
            
            return transformer

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