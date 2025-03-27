# Optimization Algorithms Comparison for Neural Networks

This assignment explores the performance of different optimization algorithms in training neural networks. We implemented and compared various optimizers to understand their impact on model convergence and accuracy.

## Optimizers Implemented

We tested the following optimization algorithms:

- AdaGrad
- AdaDelta
- Adam
- SGD with Momentum
- RMSprop

## Results

### Accuracy Plot

![Training Accuracy vs Epochs](assets/accuracy-plot.png)

### Loss Plot

![Training Loss vs Epochs](assets/loss-plot.png)

### Legend for above plots

![Legend](assets/legend.png)

## Conclusions

While all optimizers eventually reached the same loss, the Adam optimizer clearly outperforms all other optimizers in speed as well as training accuracy.
