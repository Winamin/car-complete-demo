# CAR Mathematical Specifications

## Core Definitions

### Unit State

Each autonomous computational unit is defined by a state triplet:

$$\text{Unit State}_i = \left[ A_i, v_i, \mathbf{x}_i \right]$$

Where:
- $A_i \in [0, 1]$ is the activation weight
- $v_i \in [0, 1]$ is the validation score
- $\mathbf{x}_i \in \mathbb{R}^D$ is the data sample vector

### Initial State

$$A_i^{(0)} = 0.1, \quad v_i^{(0)} = 0.5, \quad \mathbf{x}_i^{(0)} = \emptyset$$

## Tanh Transformation

The bounded activation function maps activation to [-1, 1]:

$$\tanh(A_i) = \frac{e^{A_i} - e^{-A_i}}{e^{A_i} + e^{-A_i}} \in (-1, 1)$$

Properties:
- $\tanh(0) = 0$ (neutral)
- $\tanh(1) \approx 0.76$ (positive)
- $\tanh(0.1) \approx 0.10$ (slight positive)

## Score-Based Retrieval

The retrieval score combines multiple factors:

$$s_i = A_i \cdot v_i \cdot \frac{1}{1 + \Delta_i}$$

Where:
- $\Delta_i = \|\mathbf{x}_{\text{noisy}} - \mathbf{x}_{\text{pattern}}\|$ is the deviation
- $A_i$ is activation weight
- $v_i$ is validation score

### Multi-Factor Weighting

The complete prediction weight combines multiple factors:

$$w_i = s_i \cdot v_i \cdot \log(u_i + 1) \cdot \tau_i \cdot \delta_i$$

Where:
- $s_i$: Retrieval score
- $v_i$: Confidence/validation score
- $u_i$: Usage count
- $\tau_i$: Temporal decay factor
- $\delta_i$: Diversity bonus (1 + DIVERSITY\_BONUS for special patterns)

## Consensus Formation

### Weight Normalization

$$\omega_i = \frac{w_i}{\sum_{j} w_j}$$

### Consensus Prediction

$$y_{\text{consensus}} = \sum_{i} \omega_i \cdot p_i$$

Where $p_i$ is the prediction of pattern $i$.

### Consensus Confidence

$$v_{\text{consensus}} = \frac{1}{1 + \sqrt{\text{Var}(p | \omega)} / S}$$

Where:
- $\text{Var}(p | \omega)$ is the weighted variance
- $S$ is the success threshold

## Validation Score Update

### Exponential Moving Average

$$v_i^{(t+1)} = (1 - \lambda) \cdot v_i^{(t)} + \lambda \cdot \mathbb{I}(\text{prediction correct})$$

Where $\lambda$ is the verification learning rate.

### Historical Component

$$v_i^{\text{hist}} = \frac{\sum_{\tau=1}^{t} \lambda^{t-\tau} \cdot \mathbb{I}(\text{correct}_{\tau})}{\sum_{\tau=1}^{t} \lambda^{t-\tau}}$$

## Difference Infection Dynamics

### Activation Update

$$A_i^{(t+1)} = (1 - \beta) \cdot A_i^{(t)} + \beta \cdot \frac{1}{|\mathcal{P}_i|} \sum_{j \in \mathcal{P}_i} \omega_{ij} \cdot A_j^{(t)}$$

Where:
- $\beta$ is the consensus learning rate
- $\mathcal{P}_i$ is the set of peers for unit $i$
- $\omega_{ij}$ is the weight from peer influence

### Tanh Difference

$$\Delta \tanh = |\tanh(A_i) - \tanh(A_j)|$$

- Small $\Delta \tanh$ → successful communication → converge
- Large $\Delta \tanh$ → failed communication → diverge

## Knowledge Base Operations

### Cosine Similarity

$$\text{sim}(\mathbf{x}, \mathbf{p}) = \frac{\mathbf{x} \cdot \mathbf{p}}{\|\mathbf{x}\| \cdot \|\mathbf{p}\|}$$

### Pattern Merging

$$\mathbf{p}_{\text{new}} = \alpha \cdot \mathbf{p}_{\text{old}} + (1 - \alpha) \cdot \mathbf{x}$$

$$v_{\text{new}} = \max(v_{\text{old}}, v_{\text{new}})$$

Where $\alpha$ is the merge weight.

### Temporal Decay

$$\tau_i = \frac{1}{1 + (t_{\text{now}} - t_i) \cdot 0.001}$$

## Noise Model

### Additive Gaussian Noise

$$\mathbf{x}_{\text{noisy}} = \mathbf{x} + \sigma_{\text{noise}} \cdot \xi$$

Where $\xi \sim \mathcal{N}(0, I)$.

### Noise Multiplier

$$\text{Noise Multiplier} = \frac{\sigma_{\text{noise}}}{\sigma_{\text{signal}}}$$

### Signal-to-Noise Ratio

$$\text{SNR} = 20 \log_{10}\left(\frac{\sigma_{\text{signal}}}{\sigma_{\text{noise}}}\right) = -20 \log_{10}(\text{Noise Multiplier})$$

## Float64 Precision Limits

### Maximum Representable Value

$$\text{MAX}_{\text{float64}} \approx 1.8 \times 10^{308}$$

### Overflow Condition

Overflow occurs when:

$$\Delta_i^2 > \text{MAX}_{\text{float64}}$$

This happens at noise multipliers around $10^{200}$.

### Information Recovery Ratio

$$\text{Recovery Ratio} = \frac{\|\mathbf{x}_{\text{noisy}} - \mathbf{x}\|}{\|\mathbf{x}\|} \cdot \text{Noise Multiplier}$$

Signal becomes indistinguishable at noise > $10^{15}$.

## Key Invariants

1. **Bounded Activation**: $A_i \in [0, 1]$ always
2. **Bounded Validation**: $v_i \in [0, 1]$ always
3. **Normalized Weights**: $\sum_i \omega_i = 1$
4. **Monotonic Confidence**: $v_i$ never decreases dramatically
