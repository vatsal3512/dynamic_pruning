# Detailed Project Report: Dynamic Pruning in Deep Neural Networks

## Executive Summary
This project implements an advanced model compression technique known as **Dynamic Sparsity Training** (Dynamic Pruning). Unlike traditional static pruning, which removes weights permanently after training, this approach continuously evaluates and removes less important connections *during* the training process. The project successfully validates this approach on three standard architectures (LeNet-4, AlexNet, and ResNet-18) using CIFAR-10 and MNIST datasets, achieving massive parameter reduction with minimal performance degradation.

> [!NOTE]
> The primary objective of this technique is to discover a sparse subnetwork (reminiscent of the **Lottery Ticket Hypothesis**) that requires significantly less computational memory and power while retaining the predictive capacity of the full network.

## Technical Methodology

### 1. The 2-Layer Threshold Algorithm
The core innovation in this codebase is the **2-Layer Threshold-based pruning** mechanism. It acts as a hysteresis loop for the weights to stabilize sparsity. 

For every targeted layer (Convolutional and Linear), the algorithm defines two dynamic thresholds based on the layer's current weight distribution:
- `a`: A lower bound threshold (derived from a decaying constant `k` and the maximum weight magnitude).
- `b`: An upper bound threshold (derived from `a` plus a factor of the weight's standard deviation).

The mask generation follows this logic:
1. **Strong weights** (`weights > b`): Unconditionally kept.
2. **Moderate weights** (`weights > a`): Kept *only* if they were already present in the previous iteration's mask.
3. **Weak weights** (`weights < a`): Pruned.

```python
# Core mechanism representation
temp_mask1 = (weights > b) # Strong weights
temp_mask2 = (weights > a) # Moderate weights
temp_mask = (temp_mask2 * old_mask) + temp_mask1 # Final mask
```

> [!TIP]
> **Weight Regrowth**: Because `temp_mask1` unconditionally activates any weight that grows larger than `b`, a previously pruned weight can "regrow" if gradient updates push its magnitude past the threshold. This provides the network with extreme flexibility to correct premature pruning decisions.

### 2. Annealing the Pruning Rate
Pruning doesn't happen continuously on every batch. A `prob_threshold` variable controls the probability that a pruning surgery occurs in a given training step. This probability starts at `1.0` (100% chance) and decays exponentially (`prob_threshold *= 0.999x`) over the training run.
This allows the network to aggressively prune early on, and gradually stabilize its architecture as the weights converge.

---

## Architectural Implementations

### [LeNet-4](file:///d:/Downloads/dynamic_pruning-main/dynamic_pruning-main/lenet4_prunned.py)
- **Structure**: 3 Convolutional blocks followed by 3 Fully Connected layers.
- **Purpose**: Serves as a fast, reliable baseline to validate the mathematical correctness of the custom pruning masks and data pipelines before scaling up.

### [AlexNet](file:///d:/Downloads/dynamic_pruning-main/dynamic_pruning-main/alexnet_pruning.py)
- **Structure**: 5 Convolutional layers and 3 Fully Connected layers.
- **Results**: Demonstrated extreme compression capabilities. Logs from the Jupyter Notebook indicate a reduction from **23.2 million parameters** down to **~294,000 parameters**.
- **Impact**: Achieved a staggering **79x compression rate (98.7% reduction)** without loss, proving the technique works exceptionally well on heavily over-parameterized models.

### [ResNet-18](file:///d:/Downloads/dynamic_pruning-main/dynamic_pruning-main/resnet.py)
- **Structure**: Custom implementation with `BasicBlock` and skip connections, scaling to much deeper representations.
- **Results**: Navigated the complexities of residual gradients, consistently hitting targets of **80% parameter reduction (5x compression)** on CIFAR-10 with no degradation in accuracy.

---

## Training Pipeline & Optimizations
- **Optimizers**: Adam optimizer (`lr=1e-3`, weight decay `1e-7`) used across all networks.
- **Adaptive Scheduling**: Combines `ReduceLROnPlateau` (to slash learning rates when validation loss plateaus) and `StepLR` (for a slow, continuous decay).
- **Dataset Agnosticism**: Standardized data loaders allow for seamless switching between **CIFAR-10** (RGB, 32x32) and **MNIST** (Grayscale transformed to 3-channel, 32x32) to validate that the pruning policies aren't dataset-dependent.
- **Early Stopping**: Integrated to halt training autonomously once the validation loss hits an absolute minimum limit, preventing the algorithm from over-pruning or overfitting the sparse subnetwork.

## Conclusion
The repository successfully implements a dynamic, state-of-the-art compression pipeline. By integrating threshold-based weight regrowth, it effectively solves the static pruning problem where important weights are irrecoverably deleted. The results—especially the near 99% compression on AlexNet—highlight its readiness for edge-deployment scenarios where memory footprint is a critical constraint.
