# Jpeg Reconstuction

## Summary
A module to fix jpeg compression artifacts with a small CNN, written in PyTorch.
It achieves mean error of 1.32 compared to 1.62 when comparing to the original jpegs over the test set containing the series: 0, 1, 2, 3, 4, 7.

## Contents
##### train.py
Training the model using the supplied data
##### eval.py
Test a model on the above test set and show an example of jpeg reconstruction.
##### network.py
PyTorch implementation of the network.
##### data.py
Data loading utilities
##### model.pkl
Checkpoint of the model with the reported performance to be used with the test or demo. 