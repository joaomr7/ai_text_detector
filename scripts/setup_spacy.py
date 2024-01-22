import spacy

def download_spacy_model():
    # this script installs the required spacy model
    spacy.cli.download(model='en_core_web_md')