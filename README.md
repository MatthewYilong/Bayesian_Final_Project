# Bayesian_Final_Project

## Overview

This is the final project for STAT 465 Bayesian Statistics, Spring 2024. The project provides a brief introduction to the Bayesian Neural Network, its properties, and architecture. It then demonstrates the application of the LSTM model in MINIST multi-classification task. 

## Dependencies

### Basic
- [Numpy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org)

### Deep Learning
- [PyTorch](https://pytorch.org/)
- [Pyro](https://pyro.ai) 

## Instructions

- **`lstm-intro.ipynb`**: This Jupyter notebook contains the raw outputs from our experiments as discussed in our paper.
- **`model.py`**: Constructs a Bayesian Neural Network using PyTorch and Pyro.
- **`predict.py`**: Provides various prediction methods using an ensemble approach.
- **`trainer.py`**: Contains the code for training the neural network.

### Usage

1. Train the model using `trainer.py`.
2. Evaluate the results using functions from **`predict.py`**.

## Authors

- Matthew Wu
- Yufeng Wu  

## References

This project draws inspiration from various existing implementations and research in Bayesian Statistics. Notably, the preprocessing component is inspired by the work of Jospin et al., available at: [https://arxiv.org/pdf/2007.06823.pdf](https://arxiv.org/pdf/2007.06823.pdf).
