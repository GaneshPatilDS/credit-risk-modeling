# Import necessary modules from setuptools and typing
from setuptools import find_packages, setup
from typing import List

# Function to read requirements from a file and return them as a list
def read_requirements(file_path: str) -> List[str]:
    """
    Reads the requirements from a file and returns them as a list.
    
    Args:
        file_path (str): Path to the requirements file.
    
    Returns:
        List[str]: List of requirements.
    """
    # Open the file in read mode and read all lines
    with open(file_path, 'r') as file:
        # Strip newline characters and filter out empty lines and '-e .'
        requirements = [line.strip() for line in file.readlines() if line.strip() and line.strip() != '-e .']
    
    return requirements

# Setup the package
setup(
    # Package name
    name='Credit Risk Modeling',
    # Package version
    version='0.0.1',
    author='Ganu Patil',
    author_email='ganeshpatil91803@gmail.com',
    install_requires=read_requirements('requirements.txt'),
    packages=find_packages(),   
)