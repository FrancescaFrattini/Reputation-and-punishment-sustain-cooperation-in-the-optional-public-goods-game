from setuptools import setup, find_packages

setup(
   name='opgar',
   version='2.0',
   description='Optional Public Good game And Reputation with Q-Learning',
   author='Shirsendu Podder',
   author_email='ucabpod@ucl.ac.uk',
   maintainer='Francesca Frattini',
   maintainer_email='f.frattini@unimore.it',
   packages=['opgar'],  #same as name
   install_requires=['numpy', 'pandas', 'tqdm', 'networkx'], #external packages as dependencies
)