import spacy

def main():
    # this script installs the required spacy model
    spacy.cli.download(model='en_core_web_md')

if __name__ == '__main__':
    main()