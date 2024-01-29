import os
from dataclasses import dataclass

import re

import numpy as np
import pandas as pd

import spacy
from spacy.tokens import Doc

from symspellpy import SymSpell, Verbosity
import importlib.resources

import textstat

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

    def __init__(self, nlp=None, sym_spell=None):
        self.__nlp = nlp
        self.__sym_spell = sym_spell

    def __reduce__(self):
        # do not serialize nlp and sym_spell
        return (self.__class__, (None, None))
        
    def __initialize_resources(self):
        '''
        Helper function to load the required TextPreprocessing resources.
        '''

        try:
            # check if nlp is already loaded
            if self.__nlp is None:
                # setting up spacy and disabling unused pipeline components
                self.__nlp = spacy.load('en_core_web_md', disable=['parser', 'ner', 'lemmatizer', 'textcat', 'custom'])

            # checl if sym_spell is already loaded
            if self.__sym_spell is None:
                # setting up SymSpell to typo count
                self.__sym_spell = SymSpell(max_dictionary_edit_distance=1, prefix_length=7)

                # load symspell english dictionary
                with importlib.resources.open_text('symspellpy', 'frequency_dictionary_en_82_765.txt') as file:
                    self.__sym_spell.load_dictionary(file.name, term_index=0, count_index=1)

        except Exception as e:
            raise CustomException(e)

    def __calculate_reading_ease_score(self, doc: Doc) -> int:
        '''
        Helper function to calculate reading ease score.

        Parameters
        ---
        * doc: spacy doc.

        Return
        ---
        * reading ease score.
        '''

        try:
            return textstat.flesch_reading_ease(doc.text)
        except ZeroDivisionError:
            return 0
        
    def __calculate_lexical_diversity_score(self, doc: Doc) -> float:
        '''
        Helper function to calculate lexical diversity score.

        Parameters
        ---
        * doc: spacy doc.

        Return
        ---
        * lexical diversity score.
        '''

        # get words count
        total_words_count = len(doc)
        unique_words_count = len(set(token.text.lower() for token in doc))

        if total_words_count == 0:
            return 0.0
        else:
            return unique_words_count / total_words_count
     
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        '''
        This function applies:
            * text tokenization
            * stopwords removing
            * irrelevant information removing
            * nouns removing
            * typos fixing
            * word2vec with spacy
        
        Also, this function calculate:
            * lexical diversity score
            * reading ease score

        Parameters
        ---
        * data: text data to prepare the data
        
        Returns
        ---
        * a numpy array with the columns: 300 size vector (representing the text), lexical_deversity_score and reading_ease_score
        '''

        try:
            # initialize resources
            self.__initialize_resources()

            # tranfrom texts in spacy docs (wich applies tokenization)
            docs = [doc for doc in self.__nlp.pipe(X)]

            text_vectors = []
            reading_ease_scores = []
            lexical_diversity_scores = []

            for doc in docs:

                # reading ease score
                reading_ease_scores.append(self.__calculate_reading_ease_score(doc))

                # lexical diversity score
                lexical_diversity_scores.append(self.__calculate_lexical_diversity_score(doc))

                words = []
                spaces = []
                for token in doc:
                    if token.is_stop or not re.match('[a-zA-Z]', token.text): # ignore stopwords and irrelevant information
                        continue

                    if token.pos_ in ['NOUN', 'PROPN']: # ignore nouns
                        continue

                    word = token.text.lower()

                    # check if it has vector representaion
                    if not token.has_vector:
                        # lookup for typo
                        suggestions = self.__sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=1)

                        # check if has any suggestion
                        if not suggestions:
                            continue

                        # fix typo to the closest suggestion
                        word = suggestions[0].term
                            
                    words.append(word)
                    spaces.append(token.whitespace_)

                # create and add text vector to list
                clean_doc = Doc(vocab=self.__nlp.vocab, words=words, spaces=spaces)
                text_vectors.append(clean_doc.vector)

            # return numpy array
            return np.column_stack([text_vectors, lexical_diversity_scores, reading_ease_scores])

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

    preprocessor_obj_file_path: str = ARTIFACTS_PATH / 'preprocessor.pkl'

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
                    ('scaling', MinMaxScaler())
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