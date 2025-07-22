from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='documentlm',
    description='This is a description for abc',
    version='0.1.0',
    install_requires=required,
    dependency_links=['https://github.com/facebookresearch/detectron2/archive/refs/tags/v0.6.zip'],
    packages=find_packages(where='.', exclude=['tests']),
    test_suite="tests",
)
