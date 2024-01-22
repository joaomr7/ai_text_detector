import streamlit as st

from src.pipeline.predict_pipeline import PredictPipeline
from src.components.model_trainer import MODEL_BEST_LIMIAR

MIN_DOCUMENT_CHARACTERS = 300
MAX_DOCUMNETS_CHARACTERS = 3000

# model
@st.cache_resource
def get_predict_pipeline():
    return PredictPipeline()

def predict_documnet(document):
    proba = st.session_state.pipeline.predict_single_document(document)
    return proba

# controller
def initialized_predict_pipeline():
    if 'predict_pipeline' not in st.session_state:
        st.session_state.pipeline = get_predict_pipeline()

def validate_document(document):
    return len(document.strip()) >= MIN_DOCUMENT_CHARACTERS

def process_proba(proba, best_limiar):
    '''
    Helper function to evaluate the given probability into confidence intervals.

    Paramaters
    ---
    * proba: the probability.
    * best_limiar: the decision limiar.

    Return
    ---
    * bool: generated or not by an AI.
    * str: formated confidence level.
    '''

    if proba < best_limiar:
        proba_section = best_limiar / 3

        if proba < proba_section:
            return False, '**Human document** detected with **HIGH** confidence'
        
        elif proba < proba_section * 2:
            return False, '**Human document** detected with **MEDIUM** confidence'
        
        else:
            return False, '**Human document** detected with **LOW** confidence'
        
    else:
        proba_section = (1 - best_limiar) / 3

        if proba < best_limiar + proba_section:
            return True, '**AI document** detected with **LOW** confidence'
        
        elif proba < best_limiar + proba_section * 2:
            return True, '**AI document** detected with **MEDIUM** confidence'
        
        else:
            return True, '**AI document** detected with **HIGH** confidence'

# view
def main_form():

    form = st.form(key = 'main_form', border=False)

    with form:
        # document text area
        document = st.text_area(
            label='Enter your document',
            max_chars=MAX_DOCUMNETS_CHARACTERS,
            height=300)
        
        if st.form_submit_button(label='Submit'):

            # validate document content
            if not validate_document(document):
                st.error(body=f'Document must have at least **{MIN_DOCUMENT_CHARACTERS} characters**')

            else:
                # predict proba
                documnet_proba = predict_documnet(document)
                result, label = process_proba(documnet_proba, MODEL_BEST_LIMIAR)

                if result:
                    st.error(body=label)
                else:
                    st.success(body=label)

def main_page():
    # setup page
    st.set_page_config(
        page_title='AI Snitch',
        page_icon='🤖'
    )

    # load predict pipeline
    initialized_predict_pipeline()

    # Header
    st.title(':red[AI] Snitch')
    st.text('AI text detector')

    # description
    st.caption('''This tool is not perfect. Be aware that mistakes can occur and by using this you agree that any bad use of the results from this tool is completely of your responsibility.''')

    # form
    main_form()

main_page()