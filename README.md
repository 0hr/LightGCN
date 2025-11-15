LightGCN Reimplementation and Analysis
CS584 Final Project – Fall 2024

Team Members:
Ferit Ozdaban (A20571925)
Harun Rasit Pekacar (A20607262)

This repository contains our final project for CS584, where we reimplement and analyze the LightGCN model from the SIGIR 2020 paper “LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.” LightGCN simplifies graph-based recommendation by removing nonlinearities and feature transformations, while still leveraging high-order user–item connections.

Our project goals:
• Reimplementing LightGCN from scratch in PyTorch
• Reproducing key results from the original paper
• Building and tuning strong baselines (MF-BPR, NGCF, WMF, Mult-VAE)
• Conducting extensive ablations (embedding size, K layers, regularization, sampling, layer weights, dropout)
• Profiling the model’s efficiency and scalability
• Designing at least one principled extension to LightGCN
