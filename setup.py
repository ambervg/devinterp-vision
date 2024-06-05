# setup.py
from setuptools import setup, find_packages

# Function to read the contents of the requirements.txt file
def read_requirements():
    with open('requirements.txt', 'r') as req:
        return req.read().splitlines()
    
setup(
    name="LLC-estimation-vision",
    version="0.1.0",
    description='A research project on Computer Vision (CV) models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ambervg/LLC-estimation-vision',
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires='>=3.6',
)
