from setuptools import setup, find_packages

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path):
    '''
    This function return the list of requirements for this project
    '''
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name='ai_text_generated_detection',
    version='1.0.0',
    description='project to predict whether a text was created by an A.I. or human',
    author='João Marcos Ressetti',
    author_email='jmressetti.3.4@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)