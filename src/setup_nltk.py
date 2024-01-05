import nltk

NLTK_DATA_PATH = 'nltk_data/'

def main():
    # this script installs the required nltk resources
    nltk.download('stopwords', download_dir=NLTK_DATA_PATH)
    nltk.download('punkt', download_dir=NLTK_DATA_PATH)
    nltk.download('averaged_perceptron_tagger', download_dir=NLTK_DATA_PATH)

if __name__ == '__main__':
    main()