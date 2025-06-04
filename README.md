# SVEBI: Spiking Visual Explanation By Interpretation

This repository contains the code related to the methods and experiments implemented in the paper: "SVEBI: Spiking Visual Explanation By Interpretation". 

## File descriptions

SVEBI is implemented through three different steps: 

1. collect_activation_maps.py: This file implements the proces of collecting all (average) activation maps of all filters of the model over all samples of the training set and saves them in './output/activation_maps/'. Following the terminology of the SVEBI paper, this means generating the X and L matrices.
2. identify_relevant_filters.py: This file implements the proces for identifying all relevant filters for all classes of the classification task. Following the paper, X and L are used to produce W (which indicates the relevant filters) by solving a μ-lasso optimization problem.
3. generate_explanations.ipynb: This notebook demonstrates how to use the relevant filters to generate explanatory heatmaps of an input sample to explain the model's output prediction. Generating the explanatory heatmaps follows the procedure described in the paper as well. 

The code for building and training the spiking model: Spiking VGG19 (SVGG19) can be found in the ./model directory. This directory contains code for the model definition (model.py), training (train.py) and testing (test.py). The training and testing procedure were partly inspired by the implementation details from [Revisiting Batch Normalization for Training Low-latency Deep Spiking Neural Networks from Scratch](https://arxiv.org/abs/2010.01729) and their [GitHub repository](https://github.com/Intelligent-Computing-Lab-Yale/BNTT-Batch-Normalization-Through-Time?tab=readme-ov-file).