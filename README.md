# micrograd-xor-mlp
Implementation of basic Automatic differentiation engine and XOR using MLP from scratch.
# Assignment - Week 2

## Overview

This repository contains solutions for the Week 2 assignment:

- Q1: Implementation of a basic Automatic Differentiation Engine (Micrograd)
- Q2: Implementation of XOR using a Multi-Layer Perceptron (MLP) from scratch

The objective of this assignment is to understand the core concepts behind automatic differentiation, computational graphs, backpropagation, and neural networks without relying on deep learning frameworks.

# Q1 - Automatic Differentiation Engine (Micrograd)

## Objective

Implement a basic automatic differentiation engine capable of building a computational graph and automatically computing gradients using reverse-mode automatic differentiation (backpropagation).

## Features

- Scalar-valued computational graph
- Operator overloading
- Automatic gradient computation
- Reverse-mode autodiff
- Backpropagation through computational graphs

## Concepts Learned
- Chain Rule
- Automatic Differentiation
- Gradient Computation
- Backpropagation

## Example

Given:

d = a * b + a

where:

a = 2  
b = 3

Output:

d = 8  
∂d/∂a = 4  
∂d/∂b = 2

This demonstrates how gradients are automatically calculated by traversing the computational graph in reverse order.

# Q2 - XOR using MLP from Scratch

## Objective

Implement a Multi-Layer Perceptron (MLP) from scratch to solve the XOR problem.

## Why XOR?

The XOR problem is a classic example that cannot be solved using a single perceptron because it is not linearly separable.

### XOR Truth Table

| Input A | Input B | Output |

| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

A neural network with at least one hidden layer is required to learn this non-linear relationship.

## Features

- Implemented completely from scratch
- Forward propagation
- Backpropagation
- Gradient descent optimization
- Hidden layer architecture
- Non-linear activation functions

## Training Process

1. Initialize network parameters
2. Perform forward propagation
3. Compute loss
4. Calculate gradients using backpropagation
5. Update weights using gradient descent
6. Repeat until convergence

## Expected Results

After training, the network correctly predicts:

Input: [0, 0] → 0  
Input: [0, 1] → 1  
Input: [1, 0] → 1  
Input: [1, 1] → 0

# Technologies Used

- Python 3
- Object-Oriented Programming
- Automatic Differentiation
- Neural Networks
- Backpropagation
- Gradient Descent

# Learning Outcomes

By completing this assignment, the following concepts were understood and implemented:

- Computational Graph Construction
- Reverse-Mode Automatic Differentiation
- Backpropagation Algorithm
- Gradient-Based Optimization
- Neural Network Architecture
- Multi-Layer Perceptrons
- Solving Non-Linear Classification Problems

# Conclusion

This assignment provides a practical understanding of the fundamental building blocks behind modern deep learning frameworks. The Micrograd implementation demonstrates how automatic differentiation works internally, while the XOR MLP implementation shows how neural networks learn complex, non-linear patterns through backpropagation and gradient descent.
