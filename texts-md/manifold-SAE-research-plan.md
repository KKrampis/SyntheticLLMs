# Manifold-Aware Sparse Autoencoders: A Synthetic Benchmark Approach

**Extended Technical Research Plan**

**Authors:** Research Team
**Date:** February 2026
**Based on:** SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data

---

## Abstract

Recent work has revealed that not all features in language models are one-dimensionally linear—many exist as multi-dimensional manifolds representing concepts with inherent geometric structure (Engels et al., 2025). While sparse autoencoders (SAEs) have emerged as a powerful tool for mechanistic interpretability, their ability to recover and represent features that lie on manifolds remains poorly understood. This research plan extends the SynthSAEBench framework to systematically study manifold-aware SAEs through controlled synthetic experiments.

We propose: (1) a methodology for generating synthetic activation data containing feature manifolds with known ground-truth structure, (2) novel evaluation metrics for assessing SAE performance in the presence of manifolds, (3) architectural modifications to SAEs that explicitly model manifold structure, and (4) a comprehensive experimental framework to test competing representation hypotheses. Our approach enables rigorous evaluation of whether SAEs can recover not just individual features, but the geometric relationships between them—a critical capability for understanding how language models represent structured knowledge.

**Key Contributions:**

- Extend SynthSAEBench to include circular, spherical, and toroidal feature manifolds
- Develop manifold-aware evaluation metrics beyond point-wise feature recovery
- Design SAE architectures that leverage manifold priors
- Create testable predictions comparing synthetic models to real LLM behavior

---

## 1. Introduction and Motivation

### 1.1 The Linear Representation Hypothesis and Its Limitations

The **Linear Representation Hypothesis** (LRH) posits that neural networks represent concepts as directions in activation space (Park et al., 2024b). Under this hypothesis, semantic relationships are encoded through linear operations: vector addition, subtraction, and scaling. This framework has proven remarkably successful for explaining phenomena like the classic "king - man + woman = queen" analogy.

However, recent empirical work challenges the universality of one-dimensional linear features:

**Engels et al. (2025)** demonstrate that language models contain **irreducibly multi-dimensional features**—concepts that cannot be decomposed into independent or non-co-occurring lower-dimensional components. Their key examples include:

- **Circular features:** Days of the week and months of the year form circles in activation space
- **Periodic patterns:** Temporal and cyclical concepts exhibit circular geometry
- **Intervention validation:** Steering experiments on Mistral 7B and Llama 3 8B confirm these circular structures are computationally fundamental

**Li et al. (2025)** reveal **hierarchical geometric structure** in SAE feature dictionaries across three scales:

- **Atomic scale:** "Crystal" structures with parallelogram and trapezoid faces
- **Intermediate scale:** Spatial modularity resembling functional lobes (e.g., math/code features cluster together)
- **Large scale:** Global organization reflecting semantic relationships

**Olah & Batson (2023)** introduced the **feature manifold toy model**, suggesting that related features lie on continuous manifolds where nearby points represent similar concepts. This contrasts with discrete, independent features assumed by standard SAE training.

### 1.2 The Challenge for Sparse Autoencoders

Standard SAEs are designed to recover **discrete, independent features** by enforcing sparsity through L1 regularization:

```
Loss = ||x - x̂||² + λ||f||₁

where:
  f = encoder(x)           # Sparse feature activations
  x̂ = decoder(f)           # Reconstruction
```

This objective assumes features can be represented as one-dimensional scalars in the latent space. However, **manifold features** require multiple dimensions to represent their geometric structure:

- **Days of the week:** Require 2D circular representation (not 7 independent features)
- **Colors:** Lie on a 3D manifold (hue-saturation-value)
- **Phonemes:** Form structured manifolds based on articulatory properties

**Michaud et al. (2024)** show that feature manifolds create **pathological scaling behavior**: when features lie on manifolds, SAEs allocate disproportionately many latents to "tile" high-frequency manifolds, learning far fewer distinct features than the number of latents available. This suggests fundamental architectural limitations in current SAE designs.

### 1.3 Research Questions

This research plan addresses five fundamental questions:

1. **Evaluation:** How should we evaluate SAE performance when ground-truth features are manifolds rather than independent vectors?

2. **Architecture:** What architectural modifications enable SAEs to efficiently represent manifold structure?

3. **Scaling:** How does the presence of manifolds affect SAE scaling laws and capacity allocation?

4. **Representation hypotheses:** Can synthetic models following different geometric priors (linear, circular, hierarchical) distinguish between competing theories of neural representation?

5. **LLM correspondence:** Do training dynamics and failure modes in synthetic manifold benchmarks match behaviors observed in real LLMs?

### 1.4 Why SynthSAEBench is the Right Framework

**SynthSAEBench** provides the ideal foundation for studying manifold-aware SAEs because:

1. **Ground truth control:** We define exact manifold structure and can measure recovery precisely
2. **Scalability:** Can generate millions of samples with known manifold features
3. **Ablation studies:** Isolate effects of manifold dimensionality, curvature, and density
4. **Realistic properties:** Already includes superposition, hierarchy, and correlations
5. **Validated realism:** Reproduces phenomena seen in real LLM SAEs

By extending SynthSAEBench to include manifold structure, we can systematically study how SAEs handle geometric features while maintaining the benchmark's scientific rigor.

---

## 2. Background and Related Work

### 2.1 SynthSAEBench: Foundation

SynthSAEBench generates synthetic activation data following the generative model:

```python
# 1. Sample sparse feature coefficients
c ~ TruncatedPareto(α=1, β=threshold, shape=(N,))  # N=16,384 features

# 2. Apply hierarchical constraints
c = enforce_hierarchy(c, tree_structure)

# 3. Apply low-rank correlations
c = apply_correlations(c, correlation_matrix_Σ)

# 4. Generate activation
x = D @ c + ε
where D ∈ ℝ^{d×N} is the feature dictionary (d=768 hidden dim)
      ε ~ N(0, σ²I) is Gaussian noise
```

**Key properties captured:**

- **Superposition (ρ_mm ≈ 0.15):** Features overlap in activation space
- **Sparsity (L0 ≈ 30):** Only ~30 features active per sample
- **Hierarchy:** 10,884 features in tree forest (128 roots, branching factor 4)
- **Correlations:** Low-rank structure (rank 100) models co-occurrence

This provides a **realistic but tractable** benchmark where ground truth is known.

### 2.2 Feature Manifolds: Definitions and Properties

#### 2.2.1 Formal Definition (Engels et al., 2025)

A feature is **irreducibly k-dimensional** if:

1. It activates across a k-dimensional subspace of activation space
2. It cannot be decomposed into independent lower-dimensional features
3. It cannot be decomposed into non-co-occurring lower-dimensional features

**Example:** Days of the week form a **2D circular manifold**:

- Cannot be 7 independent features (would require 7 dimensions)
- Cannot be decomposed into co-occurring 1D features (Monday through Sunday don't co-occur)
- Optimally represented as `(cos(2πt/7), sin(2πt/7))` where t ∈ {0,...,6}

#### 2.2.2 Types of Manifolds in Neural Networks

| Manifold Type  | Dimensionality | Examples                            | Parameterization                  |
| -------------- | -------------- | ----------------------------------- | --------------------------------- |
| **Linear**     | 1D             | Scalar concepts (gender, sentiment) | Single real value                 |
| **Circular**   | 2D (S¹)        | Days, months, angles                | (cos θ, sin θ)                    |
| **Spherical**  | 3D (S²)        | Directions, orientations            | (sin φ cos θ, sin φ sin θ, cos φ) |
| **Toroidal**   | 4D (S¹ × S¹)   | Periodic pairs (hour+day)           | (cos θ₁, sin θ₁, cos θ₂, sin θ₂)  |
| **Hyperbolic** | Variable       | Hierarchies, trees                  | Poincaré disk coordinates         |
| **Simplicial** | (n-1)D         | Categorical (n classes)             | Probability simplex               |

#### 2.2.3 Geometric Properties

**Curvature:** Manifolds have intrinsic curvature affecting geodesic paths

- Flat (linear subspaces): Zero curvature
- Positive (spheres): Distance between parallel geodesics decreases
- Negative (hyperbolic): Distance between parallel geodesics increases

**Density:** Feature distribution on the manifold

- Uniform: All points equally likely
- Concentrated: Clustered around specific regions
- Multi-modal: Multiple clusters on the manifold

**Noise:** Perturbations relative to manifold structure

- Tangent noise: Along the manifold (preserves manifold membership)
- Normal noise: Perpendicular to manifold (moves off the manifold)

### 2.3 SAE Architectures: Standard and Variants

#### 2.3.1 Standard L1 SAE

```python
class StandardSAE:
    def __init__(self, d_hidden, n_latents):
        self.W_enc = nn.Linear(d_hidden, n_latents)
        self.W_dec = nn.Linear(n_latents, d_hidden, bias=False)
        # Decoder columns constrained to unit norm

    def forward(self, x):
        f = F.relu(self.W_enc(x))           # Sparse latents
        x_hat = self.W_dec(f)               # Reconstruction
        return f, x_hat

    def loss(self, x):
        f, x_hat = self.forward(x)
        recon_loss = (x - x_hat).pow(2).mean()
        sparsity_loss = f.abs().mean()
        return recon_loss + self.lambda_l1 * sparsity_loss
```

**Limitations for manifolds:**

- Each latent represents a scalar (1D feature)
- No mechanism to group related latents into manifolds
- Independent sparsity penalty discourages multi-latent activation
- Cannot represent circular or spherical structure

#### 2.3.2 Gated SAE (Rajamanoharan et al., 2024a)

Separates magnitude estimation from feature activation:

```python
class GatedSAE:
    def forward(self, x):
        # Gating: which features are active (binary)
        gate_logits = self.W_gate(x)
        gate = (gate_logits > 0).float()

        # Magnitude: how much each active feature contributes
        magnitude = F.relu(self.W_mag(x))

        f = gate * magnitude               # Element-wise product
        x_hat = self.W_dec(f)
        return f, x_hat
```

**Advantages:**

- Cleaner feature selection (gating)
- Better reconstruction (separate magnitude)
- Potential for manifold extension (gate manifold groups)

#### 2.3.3 JumpReLU SAE (Rajamanoharan et al., 2024b)

Uses a discontinuous activation to improve reconstruction:

```python
def jumprelu(x, threshold=0.1):
    return torch.where(x < threshold,
                      torch.zeros_like(x),
                      x + threshold)
```

Provides better reconstruction fidelity while maintaining sparsity.

### 2.4 Evaluation Metrics Beyond Point-Wise Recovery

Current SynthSAEBench metrics assume **discrete features**:

- **MCC (Matthews Correlation):** Binary classification per feature
- **Probing F1:** Treat each latent as classifier for its matched ground-truth feature
- **Variance explained:** Reconstruction quality

**Problem:** These don't assess geometric structure recovery.

**Needed for manifolds:**

- Manifold alignment metrics
- Topology preservation measures
- Geodesic distance preservation
- Curvature estimation accuracy

We develop these in Section 4.

---

## 3. Technical Approach: Introducing Manifolds into SynthSAEBench

### 3.1 Architecture: Manifold-Structured Feature Dictionary

We extend the SynthSAEBench generative model to include **manifold features** alongside standard independent features.

#### 3.1.1 Hybrid Feature Dictionary

```python
N_independent = 12,000    # Standard 1D features
N_manifolds = 10          # Number of manifold structures
manifold_dims = [2, 2, 2, 2, 3, 3, 4, 4, 2, 2]  # Dimensions per manifold

Total feature dimensions:
  N_independent + sum(manifold_dims) = 12,000 + 26 = 12,026 effective features

Dictionary size: D ∈ ℝ^{768 × 16,384}
  - Columns 1-12,000: Independent unit vectors (as in original)
  - Columns 12,001-16,384: Grouped into 10 manifolds
```

#### 3.1.2 Manifold Generation Procedure

For each manifold m:

**Step 1: Choose manifold type and parameters**

```python
manifold_config = {
    'type': 'circular',          # or 'spherical', 'toroidal'
    'intrinsic_dim': 2,          # Dimension of manifold (k)
    'embedding_dim': 20,         # Ambient dimensions used
    'num_discretization': 32,    # Number of discrete points on manifold
    'curvature': 'constant',     # or 'variable'
    'noise_level': 0.05,        # Tangent/normal noise ratio
}
```

**Step 2: Sample base manifold in intrinsic coordinates**

For a **circular manifold** (S¹):

```python
def generate_circular_manifold(num_points, noise_level):
    # Generate evenly spaced points on circle
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)

    # Base 2D coordinates
    coords_2d = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # Add tangent noise (along the circle)
    angle_noise = np.random.randn(num_points) * noise_level
    angles_noisy = angles + angle_noise
    coords_2d = np.stack([np.cos(angles_noisy), np.sin(angles_noisy)], axis=1)

    return coords_2d, angles
```

For a **spherical manifold** (S²):

```python
def generate_spherical_manifold(num_points, noise_level):
    # Fibonacci sphere algorithm for even distribution
    indices = np.arange(num_points) + 0.5
    phi = np.arccos(1 - 2*indices/num_points)
    theta = np.pi * (1 + 5**0.5) * indices

    # 3D coordinates
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    coords_3d = np.stack([x, y, z], axis=1)

    # Add tangent noise (perpendicular to radius)
    tangent_noise = np.random.randn(num_points, 3) * noise_level
    tangent_noise -= (tangent_noise * coords_3d).sum(axis=1, keepdims=True) * coords_3d
    coords_3d += tangent_noise
    coords_3d /= np.linalg.norm(coords_3d, axis=1, keepdims=True)

    return coords_3d, (phi, theta)
```

For a **toroidal manifold** (S¹ × S¹):

```python
def generate_toroidal_manifold(num_points, major_radius, minor_radius, noise_level):
    # Sample angles uniformly
    theta = np.random.uniform(0, 2*np.pi, num_points)  # Major circle
    phi = np.random.uniform(0, 2*np.pi, num_points)    # Minor circle

    # 4D torus coordinates (or 3D embedding if desired)
    coords_4d = np.stack([
        np.cos(theta),
        np.sin(theta),
        np.cos(phi),
        np.sin(phi)
    ], axis=1)

    return coords_4d, (theta, phi)
```

**Step 3: Embed in high-dimensional space**

Map from intrinsic manifold coordinates to 768-dimensional activation space:

```python
def embed_manifold_in_activation_space(coords, embedding_dim, total_dim=768):
    """
    Embed k-dimensional manifold into d-dimensional space.

    Args:
        coords: (N, k) manifold coordinates
        embedding_dim: number of dimensions to use for embedding
        total_dim: total activation dimension (768)

    Returns:
        embedded_coords: (N, total_dim) high-dimensional coordinates
    """
    N, k = coords.shape

    # Random embedding matrix: k -> embedding_dim
    # This is a random smooth embedding
    W_embed = np.random.randn(k, embedding_dim)
    W_embed = orthogonalize_gram_schmidt(W_embed)

    # Project manifold to embedding_dim subspace
    embedded_low = coords @ W_embed  # (N, embedding_dim)

    # Place in full dimensional space
    embedded_full = np.zeros((N, total_dim))
    start_idx = np.random.randint(0, total_dim - embedding_dim)
    embedded_full[:, start_idx:start_idx+embedding_dim] = embedded_low

    # Normalize to unit norm (consistent with SynthSAEBench)
    embedded_full /= np.linalg.norm(embedded_full, axis=1, keepdims=True)

    return embedded_full
```

**Step 4: Add manifold directions to dictionary**

```python
def construct_hybrid_dictionary(N_independent, manifold_configs, d_hidden=768):
    """
    Construct feature dictionary with both independent and manifold features.
    """
    # Independent features (as in original SynthSAEBench)
    D_independent = generate_random_unit_vectors(N_independent, d_hidden)

    # Manifold features
    manifold_features = []
    manifold_metadata = []

    for i, config in enumerate(manifold_configs):
        # Generate manifold points
        if config['type'] == 'circular':
            coords, params = generate_circular_manifold(
                config['num_discretization'], config['noise_level']
            )
        elif config['type'] == 'spherical':
            coords, params = generate_spherical_manifold(
                config['num_discretization'], config['noise_level']
            )
        elif config['type'] == 'toroidal':
            coords, params = generate_toroidal_manifold(
                config['num_discretization'],
                config['major_radius'],
                config['minor_radius'],
                config['noise_level']
            )

        # Embed in activation space
        embedded = embed_manifold_in_activation_space(
            coords, config['embedding_dim'], d_hidden
        )

        manifold_features.append(embedded)
        manifold_metadata.append({
            'manifold_id': i,
            'type': config['type'],
            'intrinsic_coords': coords,
            'params': params,
            'start_idx': len(D_independent) + sum(len(m) for m in manifold_features[:-1]),
            'end_idx': len(D_independent) + sum(len(m) for m in manifold_features),
        })

    # Concatenate all features
    D_manifolds = np.vstack(manifold_features)
    D_full = np.vstack([D_independent, D_manifolds])

    return D_full, manifold_metadata
```

#### 3.1.3 Manifold Activation Statistics

Each manifold has associated statistics controlling its behavior:

```python
class ManifoldFeatureStats:
    def __init__(self, manifold_id, manifold_type, intrinsic_dim):
        self.manifold_id = manifold_id
        self.type = manifold_type
        self.intrinsic_dim = intrinsic_dim

        # Firing probability (probability this manifold is active)
        self.p_active = sample_zipfian()

        # When active, distribution over manifold surface
        self.surface_distribution = self._init_surface_distribution()

        # Magnitude distribution
        self.magnitude_mean = np.random.uniform(15.0, 25.0)
        self.magnitude_std = np.random.lognormal(0, 0.5)

    def _init_surface_distribution(self):
        """
        Define how probability mass distributes over manifold surface.
        Options:
          - 'uniform': Equal probability everywhere
          - 'concentrated': Gaussian bumps at specific locations
          - 'mixed': Multiple modes
        """
        if self.type == 'circular':
            # Could be uniform or peaked (e.g., prefer certain months)
            return UniformCircularDistribution()
        elif self.type == 'spherical':
            return UniformSphericalDistribution()
        # ... etc
```

### 3.2 Generative Model: Sampling with Manifolds

Extend the activation generation to include manifold features:

```python
def generate_activation_with_manifolds(D, manifold_metadata, feature_stats):
    """
    Generate a single activation vector with both independent and manifold features.
    """
    N_total = D.shape[1]
    N_independent = manifold_metadata[0]['start_idx']

    # Step 1: Sample independent features (as before)
    c_independent = np.zeros(N_independent)
    for i in range(N_independent):
        if np.random.rand() < feature_stats[i].p_active:
            c_independent[i] = sample_magnitude(
                feature_stats[i].magnitude_mean,
                feature_stats[i].magnitude_std
            )

    # Step 2: Sample manifold features
    c_manifolds = np.zeros(N_total - N_independent)

    for manifold_meta in manifold_metadata:
        m_id = manifold_meta['manifold_id']
        m_stats = feature_stats[N_independent + m_id]

        # Decide if this manifold is active
        if np.random.rand() < m_stats.p_active:
            # Sample a point on the manifold surface
            point_idx = m_stats.surface_distribution.sample()

            # Get the feature index corresponding to this manifold point
            feature_idx = manifold_meta['start_idx'] + point_idx - N_independent

            # Sample magnitude
            magnitude = sample_magnitude(m_stats.magnitude_mean, m_stats.magnitude_std)

            # Only ONE point on the manifold is active at a time
            # (Represents the current value of the circular/spherical feature)
            c_manifolds[feature_idx] = magnitude

            # Optional: Add smoothness by activating nearby points with decay
            if m_stats.smooth_activation:
                for neighbor_offset in [-1, 1]:
                    neighbor_idx = (point_idx + neighbor_offset) % manifold_meta['num_discretization']
                    neighbor_feature_idx = manifold_meta['start_idx'] + neighbor_idx - N_independent
                    c_manifolds[neighbor_feature_idx] = magnitude * 0.3  # Decayed activation

    # Combine independent and manifold coefficients
    c_full = np.concatenate([c_independent, c_manifolds])

    # Step 3: Apply hierarchy constraints (if any)
    c_full = enforce_hierarchy(c_full, hierarchy_tree)

    # Step 4: Apply correlations (with special handling for manifolds)
    c_full = apply_correlations_with_manifolds(c_full, correlation_matrix, manifold_metadata)

    # Step 5: Generate activation
    x = D @ c_full + np.random.randn(D.shape[0]) * noise_std

    return x, c_full, manifold_metadata
```

**Key design choices:**

1. **Discrete vs. Continuous:** We discretize manifolds (32 points on circle) to maintain compatibility with the discrete feature recovery evaluation. This is realistic—real LLMs have finite capacity.

2. **Mutual exclusivity:** For a given manifold, only one point (or smoothed neighborhood) is active per sample. This reflects that "Monday" and "Wednesday" don't co-occur as values of "day of week."

3. **Manifold-aware correlations:** Manifold features can correlate with independent features (e.g., "winter months" correlates with "cold weather").

### 3.3 Hierarchical Manifolds

Extend the tree hierarchy to include manifolds:

```
                   [Time Concept]  (root)
                  /              \
          [Cyclical]              [Linear]
          /        \                  |
    [Day of Week]  [Month]       [Timestamp]
     (circular)    (circular)      (1D continuous)
```

**Implementation:**

```python
class ManifoldHierarchy:
    def __init__(self):
        # Define hierarchical tree including manifold nodes
        self.tree = {
            'time_concept': {
                'type': 'independent',
                'children': ['cyclical', 'linear']
            },
            'cyclical': {
                'type': 'independent',
                'children': ['day_of_week', 'month']
            },
            'day_of_week': {
                'type': 'circular_manifold',
                'points': 7,
                'children': []
            },
            'month': {
                'type': 'circular_manifold',
                'points': 12,
                'children': []
            },
            'linear': {
                'type': 'independent',
                'children': ['timestamp']
            },
        }

    def enforce_constraints(self, c):
        """
        Hierarchical constraints:
        - If 'time_concept' is inactive, all children are inactive
        - If 'cyclical' is inactive, both day_of_week and month manifolds are inactive
        """
        # Implement top-down enforcement
        # ...
        return c
```

This creates **realistic structure** where manifold features participate in hierarchies, testing whether SAEs can recover both geometric and hierarchical structure simultaneously.

### 3.4 Correlation Structure with Manifolds

**Challenge:** How should manifold points correlate with each other and with independent features?

**Solution:** Extend the low-rank correlation model:

```python
def build_manifold_aware_correlation_matrix(N_independent, manifold_metadata, rank=100):
    """
    Build correlation matrix Σ that respects manifold structure.
    """
    N_total = N_independent + sum(m['num_discretization'] for m in manifold_metadata)

    # Standard low-rank correlation for independent features
    F_independent = np.random.randn(N_independent, rank) * correlation_scale

    # Manifold correlations: points on same manifold have structured correlations
    F_manifolds_list = []
    for m_meta in manifold_metadata:
        n_points = m_meta['num_discretization']

        # Create smooth correlation structure along manifold
        # Nearby points have higher correlation
        F_manifold = np.zeros((n_points, rank))

        # Use a subset of rank dimensions for this manifold
        manifold_rank_dims = np.random.choice(rank, size=min(10, rank), replace=False)

        for i, point_idx in enumerate(range(n_points)):
            # Points on manifold share some factors
            # Magnitude varies smoothly (e.g., cosine pattern along circle)
            angle = 2 * np.pi * point_idx / n_points
            for j, rank_dim in enumerate(manifold_rank_dims):
                # Smooth variation along manifold
                F_manifold[point_idx, rank_dim] = np.cos(angle + j * np.pi / len(manifold_rank_dims))

        F_manifolds_list.append(F_manifold)

    F_manifolds = np.vstack(F_manifolds_list)

    # Combine
    F_full = np.vstack([F_independent, F_manifolds])

    # Compute correlation matrix
    Sigma = F_full @ F_full.T

    # Normalize to correlation matrix (diagonal = 1)
    delta = 1 - np.diag(Sigma)
    Sigma = Sigma + np.diag(delta)

    return Sigma, F_full
```

**Effect:** Creates realistic correlation patterns where:

- Points on the same manifold are correlated
- Adjacent points on manifolds are more correlated than distant points
- Manifolds can correlate with independent features
- Overall maintains low-rank structure for efficient sampling

---

## 4. Evaluation Methodology: Assessing Manifold Recovery

Standard metrics (MCC, F1) assess **per-feature binary classification**. For manifolds, we need **geometric structure recovery metrics**.

### 4.1 Manifold Detection: Does the SAE Group Latents?

**Goal:** Detect whether the SAE has learned to represent a manifold using multiple latents.

**Method 1: Latent Co-activation Clustering**

```python
def detect_manifold_clusters(sae_activations, threshold_correlation=0.3):
    """
    Identify groups of SAE latents that consistently co-activate,
    suggesting they jointly represent a manifold.

    Args:
        sae_activations: (n_samples, n_latents) binary activation matrix

    Returns:
        clusters: List of latent groups forming potential manifolds
    """
    # Compute co-activation correlation matrix
    # Corr[i,j] = frequency that latent i and j are both active
    co_activation = (sae_activations.T @ sae_activations) / sae_activations.shape[0]

    # Threshold to adjacency matrix
    adjacency = (co_activation > threshold_correlation).astype(float)
    np.fill_diagonal(adjacency, 0)

    # Find connected components (potential manifold groups)
    from scipy.sparse.csgraph import connected_components
    n_clusters, labels = connected_components(adjacency, directed=False)

    clusters = []
    for cluster_id in range(n_clusters):
        latent_group = np.where(labels == cluster_id)[0]
        if len(latent_group) >= 2:  # At least 2D for a manifold
            clusters.append(latent_group)

    return clusters
```

**Method 2: Decoder Weight Geometry**

```python
def analyze_decoder_subspace_geometry(decoder_weights, latent_group):
    """
    Analyze geometric structure of decoder columns for a latent group.

    If latents form a manifold, their decoder columns should span
    a low-dimensional subspace with manifold structure.
    """
    # Extract decoder columns for this latent group
    W_dec_group = decoder_weights[:, latent_group]  # (d_hidden, k)

    # Perform PCA to find intrinsic dimensionality
    U, S, Vt = np.linalg.svd(W_dec_group, full_matrices=False)

    # Intrinsic dimensionality = number of significant singular values
    intrinsic_dim = np.sum(S > 0.1 * S[0])

    # Project decoder columns to top-k principal components
    W_dec_projected = Vt[:intrinsic_dim, :].T  # (n_latents, intrinsic_dim)

    # Analyze geometry in projected space
    if intrinsic_dim == 2:
        # Check for circular structure
        circularity_score = measure_circularity(W_dec_projected)
    elif intrinsic_dim == 3:
        # Check for spherical structure
        sphericity_score = measure_sphericity(W_dec_projected)

    return {
        'intrinsic_dim': intrinsic_dim,
        'singular_values': S,
        'geometry_type': infer_geometry_type(W_dec_projected),
        'geometry_score': compute_geometry_score(W_dec_projected),
    }

def measure_circularity(points_2d):
    """
    Measure how circular a set of 2D points is.
    """
    # Fit circle: find center and radius minimizing distance to points
    from scipy.optimize import least_squares

    def circle_residuals(params, points):
        cx, cy, r = params
        return np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2) - r

    initial_guess = [points_2d[:, 0].mean(), points_2d[:, 1].mean(), 1.0]
    result = least_squares(circle_residuals, initial_guess, args=(points_2d,))

    # Circularity = 1 - (std of residuals) / radius
    cx, cy, r = result.x
    residuals = circle_residuals(result.x, points_2d)
    circularity = 1 - (np.std(residuals) / r)

    return circularity
```

### 4.2 Manifold Alignment Score

**Goal:** Measure how well recovered manifold aligns with ground-truth manifold.

**Metric: Geodesic Distance Preservation**

```python
def manifold_alignment_score(gt_manifold_points, sae_latent_group, sae_activations):
    """
    Measure alignment between ground-truth manifold and SAE-learned representation.

    Idea: Geodesic distances on the manifold should be preserved
    in the SAE latent space.

    Args:
        gt_manifold_points: (N, k) ground-truth manifold coordinates
        sae_latent_group: indices of SAE latents representing this manifold
        sae_activations: (n_samples, n_latents) SAE activations

    Returns:
        alignment_score: 0-1, higher is better
    """
    # Extract activations for this latent group
    group_activations = sae_activations[:, sae_latent_group]  # (n_samples, k')

    # Compute pairwise geodesic distances on ground-truth manifold
    D_gt = compute_geodesic_distances(gt_manifold_points)

    # Compute pairwise Euclidean distances in SAE latent space
    D_sae = pairwise_distances(group_activations)

    # Measure correlation between distance matrices (Mantel test / Procrustes)
    # Good alignment → monotonic relationship between D_gt and D_sae
    from scipy.stats import spearmanr

    # Flatten upper triangular parts
    gt_dists = D_gt[np.triu_indices_from(D_gt, k=1)]
    sae_dists = D_sae[np.triu_indices_from(D_sae, k=1)]

    correlation, p_value = spearmanr(gt_dists, sae_dists)

    return max(0, correlation)  # Clip to [0, 1]

def compute_geodesic_distances(manifold_points):
    """
    Compute geodesic distances for different manifold types.
    """
    if manifold_type == 'circular':
        # For circle: geodesic distance = arc length
        # If points are at angles θ_i, θ_j, geodesic distance = min(|θ_i - θ_j|, 2π - |θ_i - θ_j|)
        angles = manifold_points['angles']
        N = len(angles)
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                diff = abs(angles[i] - angles[j])
                D[i, j] = D[j, i] = min(diff, 2*np.pi - diff)
        return D

    elif manifold_type == 'spherical':
        # For sphere: geodesic distance = great circle distance
        # d(p, q) = arccos(p · q) where p, q are unit vectors
        points = manifold_points['coords_3d']  # Already normalized
        D = np.arccos(np.clip(points @ points.T, -1, 1))
        return D

    # ... other manifold types
```

### 4.3 Topology Preservation: Persistent Homology

**Goal:** Verify that topological properties (e.g., circular loops, spherical shells) are preserved.

```python
def topological_alignment(gt_manifold_points, sae_latent_activations):
    """
    Use persistent homology to compare topology of ground-truth vs. learned manifold.
    """
    from ripser import ripser
    from persim import plot_diagrams

    # Compute persistence diagrams
    dgm_gt = ripser(gt_manifold_points)['dgms']
    dgm_sae = ripser(sae_latent_activations)['dgms']

    # Compare H1 (1-dimensional holes, i.e., circular loops)
    # For a circular manifold, should have 1 persistent loop
    h1_gt = dgm_gt[1]  # (birth, death) pairs for 1-cycles
    h1_sae = dgm_sae[1]

    # Measure bottleneck distance between persistence diagrams
    from persim import bottleneck
    distance_h1 = bottleneck(h1_gt, h1_sae)

    # Lower distance → better topological preservation
    topology_score = np.exp(-distance_h1)

    return topology_score
```

### 4.4 Per-Point Feature Recovery: Adapted MCC

Extend MCC to handle manifold points:

```python
def manifold_aware_mcc(gt_coefficients, sae_activations, manifold_metadata):
    """
    Compute MCC for manifold features.

    Challenge: Multiple SAE latents may jointly represent a manifold point.
    Solution: Assign each ground-truth manifold point to the best-matching
              SAE latent within the detected manifold cluster.
    """
    scores = []

    for manifold_meta in manifold_metadata:
        m_start = manifold_meta['start_idx']
        m_end = manifold_meta['end_idx']

        # Ground-truth activations for this manifold
        gt_manifold = (gt_coefficients[:, m_start:m_end] > 0).astype(int)

        # Detect SAE latent cluster representing this manifold
        sae_cluster = detect_best_matching_cluster(sae_activations, gt_manifold)

        if sae_cluster is None:
            # SAE failed to learn this manifold
            scores.append(0.0)
            continue

        # For each ground-truth point, find best matching SAE latent in cluster
        sae_cluster_activations = sae_activations[:, sae_cluster]

        # Compute MCC for each GT point vs each SAE latent
        mcc_matrix = np.zeros((gt_manifold.shape[1], len(sae_cluster)))
        for i in range(gt_manifold.shape[1]):
            for j in range(len(sae_cluster)):
                mcc_matrix[i, j] = matthews_corrcoef(
                    gt_manifold[:, i],
                    sae_cluster_activations[:, j]
                )

        # Optimal bipartite matching (Hungarian algorithm)
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-mcc_matrix)

        # Average MCC over matched pairs
        manifold_mcc = mcc_matrix[row_ind, col_ind].mean()
        scores.append(manifold_mcc)

    return np.mean(scores)
```

### 4.5 Curvature Estimation Accuracy

**Goal:** Assess whether SAE captures the intrinsic curvature of the manifold.

```python
def curvature_estimation_accuracy(gt_manifold, sae_latent_group, sae_activations):
    """
    Compare estimated curvature of learned manifold vs. ground truth.
    """
    # For ground-truth manifold, compute intrinsic curvature
    if gt_manifold['type'] == 'circular':
        gt_curvature = 1.0 / gt_manifold['radius']  # Curvature of circle
    elif gt_manifold['type'] == 'spherical':
        gt_curvature = 1.0 / gt_manifold['radius']  # Gaussian curvature of sphere

    # For learned SAE manifold, estimate curvature from data
    group_activations = sae_activations[:, sae_latent_group]

    # Fit quadratic form to local neighborhoods
    estimated_curvature = estimate_local_curvature(group_activations)

    # Compare
    curvature_error = abs(gt_curvature - estimated_curvature) / gt_curvature

    return 1 - curvature_error  # Score in [0, 1]
```

### 4.6 Comprehensive Evaluation Suite

Combine all metrics into a unified benchmark:

```python
class ManifoldSAEBenchmark:
    def evaluate(self, sae, dataset_with_manifolds):
        results = {
            # Standard metrics (for independent features)
            'independent_mcc': compute_mcc_independent_features(...),
            'independent_f1': compute_f1_independent_features(...),
            'variance_explained': compute_variance_explained(...),
            'l0': compute_l0_sparsity(...),

            # Manifold-specific metrics
            'manifold_detection_rate': fraction_of_manifolds_detected(...),
            'manifold_alignment_score': average_geodesic_preservation(...),
            'topology_preservation_score': average_persistent_homology_match(...),
            'manifold_mcc': manifold_aware_mcc(...),
            'curvature_accuracy': average_curvature_estimation(...),

            # Combined score
            'overall_manifold_score': weighted_average([...]),
        }

        return results
```

---

## 5. Manifold-Aware SAE Architectures

Standard SAEs learn independent latents. We propose architectural modifications that explicitly model manifold structure.

### 5.1 Grouped Latent SAE (GL-SAE)

**Idea:** Organize latents into predefined groups, each representing a potential manifold.

```python
class GroupedLatentSAE(nn.Module):
    def __init__(self, d_hidden, n_latents, n_groups, latents_per_group):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_latents = n_latents
        self.n_groups = n_groups
        self.latents_per_group = latents_per_group

        # Encoder: shared + group-specific
        self.W_enc_shared = nn.Linear(d_hidden, d_hidden)
        self.group_encoders = nn.ModuleList([
            nn.Linear(d_hidden, latents_per_group) for _ in range(n_groups)
        ])

        # Decoder: standard
        self.W_dec = nn.Linear(n_latents, d_hidden, bias=False)

        # Group gating: which groups are active?
        self.group_gate = nn.Linear(d_hidden, n_groups)

    def forward(self, x):
        # Shared representation
        h = F.relu(self.W_enc_shared(x))

        # Group gating (select which groups are active)
        group_logits = self.group_gate(x)
        group_probs = torch.sigmoid(group_logits)

        # Encode within each group
        group_features = []
        for i, enc in enumerate(self.group_encoders):
            f_group = F.relu(enc(h)) * group_probs[:, i:i+1]  # Gated
            group_features.append(f_group)

        # Concatenate all group features
        f = torch.cat(group_features, dim=1)

        # Reconstruct
        x_hat = self.W_dec(f)

        return f, x_hat

    def loss(self, x):
        f, x_hat = self.forward(x)

        recon_loss = (x - x_hat).pow(2).mean()

        # Sparsity: L1 on groups (encourage few active groups)
        group_l0 = (f.reshape(-1, self.n_groups, self.latents_per_group).abs().sum(dim=2) > 0).float().sum(dim=1).mean()

        # Sparsity: L1 within groups
        feature_l1 = f.abs().mean()

        return recon_loss + self.lambda_group * group_l0 + self.lambda_feature * feature_l1
```

**Advantages:**

- Groups can specialize to different manifolds
- Group gating provides sparsity at manifold level
- Within-group features can represent points on manifold

### 5.2 Manifold-Parametric SAE (MP-SAE)

**Idea:** Explicitly parameterize latents as manifold coordinates.

```python
class ManifoldParametricSAE(nn.Module):
    def __init__(self, d_hidden, manifold_configs):
        super().__init__()
        self.d_hidden = d_hidden
        self.manifolds = nn.ModuleList()

        for config in manifold_configs:
            if config['type'] == 'circular':
                self.manifolds.append(CircularManifoldModule(d_hidden))
            elif config['type'] == 'spherical':
                self.manifolds.append(SphericalManifoldModule(d_hidden))
            # etc.

        # Also include standard independent latents
        self.independent_sae = StandardSAE(d_hidden, n_independent_latents)

    def forward(self, x):
        # Independent features
        f_ind, x_hat_ind = self.independent_sae(x)

        # Manifold features
        manifold_outputs = []
        for manifold_module in self.manifolds:
            manifold_out = manifold_module(x)
            manifold_outputs.append(manifold_out)

        # Combine reconstructions
        x_hat = x_hat_ind + sum(m['reconstruction'] for m in manifold_outputs)

        return {
            'independent_features': f_ind,
            'manifold_features': manifold_outputs,
            'reconstruction': x_hat,
        }

class CircularManifoldModule(nn.Module):
    def __init__(self, d_hidden, embedding_dim=16):
        super().__init__()
        # Predict: is this circular feature active? If so, what angle?
        self.gate = nn.Linear(d_hidden, 1)
        self.angle_predictor = nn.Linear(d_hidden, 2)  # (cos θ, sin θ)
        self.magnitude_predictor = nn.Linear(d_hidden, 1)

        # Decoder: from angle to reconstruction
        # Parameterized as: reconstruction = magnitude * D_manifold @ [cos θ, sin θ]
        self.D_manifold = nn.Linear(2, d_hidden, bias=False)

    def forward(self, x):
        # Gate: is this manifold active?
        gate_logit = self.gate(x)
        gate = torch.sigmoid(gate_logit)

        # Angle: predict (cos θ, sin θ)
        angle_raw = self.angle_predictor(x)
        angle_normalized = F.normalize(angle_raw, dim=1)  # Project to unit circle

        # Magnitude
        magnitude = F.relu(self.magnitude_predictor(x))

        # Reconstruction contribution
        recon_contribution = self.D_manifold(angle_normalized * magnitude * gate)

        return {
            'gate': gate,
            'angle': angle_normalized,  # (cos θ, sin θ)
            'magnitude': magnitude,
            'reconstruction': recon_contribution,
        }
```

**Advantages:**

- Explicitly models manifold geometry
- Learns smooth decoder from manifold parameters
- Naturally handles continuous manifold traversal

**Challenges:**

- Requires knowing manifold type a priori
- More complex architecture

### 5.3 Hierarchical Manifold SAE (HM-SAE)

Combines hierarchical structure with manifolds:

```python
class HierarchicalManifoldSAE(nn.Module):
    """
    Encode both hierarchical and manifold structure.

    Example:
               Time (root, independent)
                  /            \
           Cyclical           Linear
           /      \              |
      DayOfWeek  Month      Timestamp
      (circular) (circular)   (1D)
    """
    def __init__(self, d_hidden, hierarchy_config):
        super().__init__()
        self.hierarchy = self._build_hierarchy(hierarchy_config)

    def forward(self, x):
        # Traverse hierarchy top-down
        # Parent features gate children
        results = self._traverse_hierarchy(x, self.hierarchy.root)
        return results

    def _traverse_hierarchy(self, x, node):
        # Compute this node's feature
        if node.type == 'independent':
            gate, magnitude = node.encoder(x)
        elif node.type == 'circular_manifold':
            gate, angle, magnitude = node.manifold_encoder(x)

        # If this node is inactive, all children are inactive
        if gate < threshold:
            return {'active': False, 'features': None}

        # Otherwise, recurse to children
        child_results = []
        for child in node.children:
            child_result = self._traverse_hierarchy(x, child)
            if child_result['active']:
                child_results.append(child_result)

        return {
            'active': True,
            'node_id': node.id,
            'features': {'gate': gate, ...},
            'children': child_results,
        }
```

### 5.4 Adaptive Latent Manifold SAE (ALM-SAE)

**Idea:** Don't assume manifold structure—let the SAE discover it.

```python
class AdaptiveLatentManifoldSAE(nn.Module):
    """
    Learns to group latents into manifolds automatically.
    Uses a learned adjacency matrix to define manifold neighborhoods.
    """
    def __init__(self, d_hidden, n_latents):
        super().__init__()
        self.encoder = nn.Linear(d_hidden, n_latents)
        self.decoder = nn.Linear(n_latents, d_hidden, bias=False)

        # Learnable adjacency: which latents are neighbors on a manifold?
        # A[i,j] = 1 if latents i and j are on the same manifold
        self.adjacency_logits = nn.Parameter(torch.randn(n_latents, n_latents))

    def forward(self, x):
        f_raw = F.relu(self.encoder(x))

        # Apply manifold smoothness: latents on same manifold should have similar activations
        adjacency = torch.sigmoid(self.adjacency_logits)
        adjacency = (adjacency + adjacency.T) / 2  # Symmetrize

        # Smooth features according to adjacency
        # (Manifold assumption: nearby points have similar representations)
        f_smoothed = f_raw + 0.1 * (adjacency @ f_raw.T).T / adjacency.sum(dim=1, keepdim=True).T

        x_hat = self.decoder(f_smoothed)
        return f_raw, f_smoothed, x_hat

    def loss(self, x):
        f_raw, f_smoothed, x_hat = self.forward(x)

        recon_loss = (x - x_hat).pow(2).mean()
        sparsity_loss = f_raw.abs().mean()

        # Manifold regularization: encourage adjacency to be sparse and block-diagonal
        # (Each manifold forms a connected component)
        adjacency = torch.sigmoid(self.adjacency_logits)
        manifold_reg = adjacency.sum() / (self.n_latents ** 2)  # Sparsity

        return recon_loss + self.lambda_l1 * sparsity_loss + self.lambda_manifold * manifold_reg
```

**Advantages:**

- Discovers manifold structure from data
- Doesn't require specifying manifold types
- Adjacency matrix is interpretable

---

## 6. Synthetic Model Generation Following Representation Hypotheses

To test competing theories of neural representation, we generate synthetic models instantiating different hypotheses, then compare SAE behavior on these models to behavior on real LLMs.

### 6.1 Representation Hypotheses to Test

| Hypothesis                                 | Description                                    | Prediction for SAEs                                                   |
| ------------------------------------------ | ---------------------------------------------- | --------------------------------------------------------------------- |
| **Linear Representation Hypothesis (LRH)** | All concepts are 1D directions                 | SAEs perfectly recover all features with sufficient capacity          |
| **Manifold Hypothesis**                    | Some concepts lie on low-dimensional manifolds | SAEs struggle unless manifold-aware; show specific failure modes      |
| **Superposition + Manifolds**              | Manifolds are in superposition                 | SAEs must handle both manifold geometry and feature interference      |
| **Hierarchical Manifolds**                 | Manifolds participate in hierarchies           | SAE feature recovery depends on correctly identifying parent features |
| **Compositional Manifolds**                | Manifolds can be composed (e.g., S¹ × S¹)      | SAEs either learn composition or tile it with many latents            |

### 6.2 Generating Synthetic Models for Each Hypothesis

#### 6.2.1 Pure LRH Model (Baseline)

```python
def generate_pure_lrh_model():
    """Original SynthSAEBench: all features are independent 1D."""
    config = {
        'N_features': 16384,
        'd_hidden': 768,
        'manifolds': [],  # No manifolds
        'superposition': 0.15,
        'hierarchy': True,
        'correlations': True,
    }
    return SynthSAEBenchDataset(config)
```

**Expected SAE behavior:**

- High MCC (~0.75) at L0=30
- Probing F1 ~0.85
- Standard precision-recall tradeoff

#### 6.2.2 Manifold Hypothesis Model

```python
def generate_manifold_hypothesis_model():
    """Replace some independent features with circular and spherical manifolds."""
    config = {
        'N_independent': 12000,
        'd_hidden': 768,
        'manifolds': [
            {'type': 'circular', 'n_points': 32, 'embedding_dim': 16, 'label': 'temporal_cycle'},
            {'type': 'circular', 'n_points': 32, 'embedding_dim': 16, 'label': 'periodic_pattern'},
            {'type': 'spherical', 'n_points': 64, 'embedding_dim': 20, 'label': 'direction'},
            {'type': 'toroidal', 'n_points': (8, 8), 'embedding_dim': 24, 'label': 'hour_day'},
            # ... more manifolds
        ],
        'superposition': 0.15,
        'hierarchy': False,  # Isolate manifold effects first
        'correlations': True,
    }
    return ManifoldSynthSAEBenchDataset(config)
```

**Expected SAE behavior:**

- Standard SAE: Low manifold alignment score (<0.3)
- Standard SAE: May "tile" manifolds (learn many latents per manifold)
- Standard SAE: Poor topology preservation
- Manifold-aware SAE: Higher manifold alignment (>0.7)
- Manifold-aware SAE: Efficient representation (2-3 latents per circular manifold)

#### 6.2.3 Superposition + Manifolds Model

```python
def generate_superposition_manifolds_model(superposition_level):
    """Test interaction between manifold structure and superposition."""
    config = {
        'N_independent': 10000,
        'manifolds': [
            {'type': 'circular', 'n_points': 32, ...},
            # ... 10 manifolds
        ],
        'superposition': superposition_level,  # Vary: 0.05, 0.15, 0.25
        'hierarchy': False,
        'correlations': True,
    }
    return ManifoldSynthSAEBenchDataset(config)
```

**Experiment:** Sweep superposition from 0.05 to 0.30.

**Expected behavior:**

- Michaud et al. (2024) predict pathological scaling when manifolds + superposition
- SAEs allocate too many latents to high-frequency manifolds
- Manifold recovery degrades faster than independent feature recovery

#### 6.2.4 Hierarchical Manifolds Model

```python
def generate_hierarchical_manifolds_model():
    """Manifolds as nodes in hierarchy."""
    config = {
        'N_independent': 8000,
        'manifolds': [
            {'type': 'circular', 'parent': 'temporal', ...},
            {'type': 'circular', 'parent': 'temporal', ...},
        ],
        'hierarchy': {
            'root': ['temporal', 'spatial', 'abstract'],
            'temporal': {
                'children': ['day_of_week_manifold', 'month_manifold'],
                'mutex': False,
            },
            # ...
        },
        'superposition': 0.15,
    }
    return HierarchicalManifoldDataset(config)
```

**Expected behavior:**

- Hierarchical SAE: Must activate parent to activate manifold child
- Errors propagate: Missing parent → miss entire manifold
- Hierarchy + manifold metrics both important

#### 6.2.5 Compositional Manifolds Model

```python
def generate_compositional_manifolds_model():
    """Test composed manifolds like S¹ × S¹ (torus)."""
    config = {
        'N_independent': 12000,
        'manifolds': [
            # Atomic manifolds
            {'type': 'circular', 'id': 'day'},
            {'type': 'circular', 'id': 'hour'},

            # Composed manifold
            {'type': 'toroidal', 'composition': ('day', 'hour'), ...},
        ],
        'composition_type': 'product',  # or 'sum', 'concat'
    }
    return CompositionalManifoldDataset(config)
```

**Expected behavior:**

- Standard SAE: Treats torus as 64 independent points (8×8 grid)
- Compositional SAE: Learns factorized representation (day latents × hour latents)
- Efficiency: Compositional uses O(n+m) latents vs O(n×m) for standard

### 6.3 Training Dynamics Comparison

Compare training dynamics on synthetic models to known LLM SAE behaviors:

```python
def compare_training_dynamics(synthetic_model, sae_architecture):
    """
    Track metrics during training:
      - Loss curves
      - Feature emergence patterns
      - Dead neuron rates
      - Manifold detection over time
    """
    results = {
        'steps': [],
        'loss': [],
        'mcc_independent': [],
        'manifold_alignment': [],
        'dead_neurons': [],
        'feature_splitting_events': [],
    }

    for step in range(num_training_steps):
        # Train step
        batch = synthetic_model.sample_batch(batch_size)
        loss = sae.train_step(batch)

        # Evaluate
        if step % eval_interval == 0:
            eval_results = evaluate_on_held_out(sae, synthetic_model)
            results['steps'].append(step)
            results['loss'].append(loss)
            results['mcc_independent'].append(eval_results['mcc'])
            results['manifold_alignment'].append(eval_results['manifold_score'])
            results['dead_neurons'].append(count_dead_neurons(sae))

            # Detect feature splitting (a known phenomenon in LLM SAEs)
            splitting = detect_feature_splitting(sae, previous_decoder)
            results['feature_splitting_events'].append(splitting)

        previous_decoder = sae.W_dec.clone()

    return results
```

**Key phenomena to reproduce from LLM SAEs:**

1. **Feature splitting** (Chanin et al., 2024): Single ground-truth feature learned by multiple SAE latents
   
   - Predict: More common for manifold features (SAE splits manifold into tiles)

2. **Feature absorption** (Chanin et al., 2024): Multiple ground-truth features collapse to one SAE latent
   
   - Predict: Less common for manifold features (manifolds resist collapse)

3. **Dead neurons**: Latents that never activate
   
   - Predict: Manifold-aware architectures have fewer dead neurons

4. **Scaling laws**: How metrics scale with latent count N
   
   - Predict: Manifold models show different scaling than pure LRH models

### 6.4 Creating Testable Predictions

Formalize predictions that distinguish hypotheses:

| Prediction                | LRH Model                  | Manifold Model                          | Test                                 |
| ------------------------- | -------------------------- | --------------------------------------- | ------------------------------------ |
| **Scaling exponent**      | MCC ~ N^α, α ≈ 0.8         | MCC ~ N^β, β < α due to manifold tiling | Fit power laws                       |
| **Latents per feature**   | 1.2 latents per GT feature | 5+ latents per manifold                 | Count matched latents                |
| **Topology preservation** | N/A (no topology)          | PH score > 0.7 for manifold-aware SAE   | Persistent homology                  |
| **Dead neuron rate**      | ~10% at convergence        | ~20% for standard SAE on manifolds      | Count inactive latents               |
| **Curvature estimation**  | N/A                        | Error < 20% for manifold-aware SAE      | Compare estimated vs. true curvature |

**Validation against LLMs:**

Compare these predictions to known behaviors in real LLM SAEs:

- Engels et al. (2025): Found circular features → expect similar detection patterns
- Li et al. (2025): Found geometric structure → expect similar decoder weight geometry
- Gao et al. (2025): Scaling laws → compare to our manifold scaling laws

---

## 7. Testing Validity of Manifold Hypotheses

### 7.1 Cross-Model Consistency

If manifolds are fundamental, SAEs trained on different models of the same hypothesis should learn similar structures.

```python
def test_cross_model_consistency():
    """
    Train 5 independent synthetic models with same manifold structure.
    Train SAE on each.
    Check if learned manifolds are consistent.
    """
    models = [generate_manifold_hypothesis_model(seed=i) for i in range(5)]
    saes = [train_sae(model) for model in models]

    # Extract learned manifold structures
    manifolds_per_sae = [detect_manifolds(sae) for sae in saes]

    # Measure consistency: do all SAEs detect similar manifold geometries?
    consistency_score = compute_manifold_consistency(manifolds_per_sae)

    # Expected: High consistency if manifolds are real structure
    #           Low consistency if manifolds are spurious
    return consistency_score
```

### 7.2 Transfer Learning Experiments

Test if manifold-aware SAEs trained on synthetic data transfer to real LLMs:

```python
def test_manifold_transfer():
    """
    1. Train manifold-aware SAE on synthetic model with known circular features
    2. Fine-tune on real LLM activations
    3. Check if SAE preferentially learns known circular features (days, months)
    """
    # Pre-train on synthetic
    synthetic_model = generate_manifold_hypothesis_model()
    sae = ManifoldParametricSAE(...)
    pretrain_sae(sae, synthetic_model)

    # Fine-tune on real LLM
    llm = load_llm('gpt2-small')
    llm_activations = collect_activations(llm, dataset)
    finetune_sae(sae, llm_activations)

    # Evaluate: Does SAE find known circular features?
    probes = {
        'day_of_week': probe_for_circular_feature(sae, days_dataset),
        'month': probe_for_circular_feature(sae, months_dataset),
    }

    # Compare to SAE trained from scratch on LLM
    baseline_sae = StandardSAE(...)
    train_sae(baseline_sae, llm_activations)
    baseline_probes = probe_baseline(baseline_sae)

    return {
        'manifold_aware': probes,
        'baseline': baseline_probes,
        'improvement': probes - baseline_probes,
    }
```

### 7.3 Ablation Studies

Systematically ablate components to understand necessity:

```python
def ablation_study():
    results = {}

    # Baseline: LRH model
    results['lrh_only'] = train_and_evaluate(generate_pure_lrh_model())

    # Add manifolds one at a time
    results['lrh + 1_circular'] = train_and_evaluate(generate_model_with_n_manifolds(1, 'circular'))
    results['lrh + 5_circular'] = train_and_evaluate(generate_model_with_n_manifolds(5, 'circular'))
    results['lrh + 10_circular'] = train_and_evaluate(generate_model_with_n_manifolds(10, 'circular'))

    # Add different manifold types
    results['lrh + spherical'] = train_and_evaluate(generate_model_with_manifold_type('spherical'))
    results['lrh + toroidal'] = train_and_evaluate(generate_model_with_manifold_type('toroidal'))

    # Vary manifold parameters
    for embedding_dim in [8, 16, 32, 64]:
        results[f'embedding_dim_{embedding_dim}'] = train_and_evaluate(
            generate_model_with_embedding_dim(embedding_dim)
        )

    return results
```

### 7.4 Causal Interventions on Manifolds

Perform interventions to test whether learned manifolds are causal:

```python
def test_manifold_causality(sae, manifold_id):
    """
    If SAE learned a circular manifold representing 'day of week':
      1. Intervene: Set manifold to 'Monday' encoding
      2. Decode to activation space
      3. Feed to downstream tasks
      4. Check if behavior changes consistently with 'Monday'
    """
    # Identify learned manifold for 'day of week'
    dow_manifold = sae.manifolds[manifold_id]

    results = {}
    for day in ['Monday', 'Tuesday', ..., 'Sunday']:
        # Set manifold to represent this day
        manifold_encoding = encode_day_on_manifold(day, dow_manifold)

        # Decode to activation
        intervened_activation = sae.decode(manifold_encoding)

        # Test on downstream task (e.g., next-word prediction)
        llm_output = llm.forward(input_with_activation=intervened_activation)

        # Check if output is day-consistent
        day_consistency = measure_day_consistency(llm_output, expected_day=day)
        results[day] = day_consistency

    # Expected: High consistency → manifold is causally meaningful
    return results
```

### 7.5 Comparing to Real LLM Behavior

Reproduce known LLM phenomena in synthetic models:

```python
def validate_against_llm_phenomena():
    """
    Known phenomena from LLM SAE literature:
    1. Calendar features form geometric structure (Leask et al., 2024)
    2. Circular features for days/months (Engels et al., 2025)
    3. Hierarchical organization (Li et al., 2025)

    Test: Do our synthetic models with manifolds reproduce these?
    """
    # Generate synthetic model with calendar manifolds
    synthetic_model = generate_calendar_manifold_model()

    # Train SAE
    sae = train_sae(synthetic_model)

    # Test 1: Geometric structure (Leask et al., 2024)
    # Expected: Day and month features form 2D structure
    calendar_geometry = analyze_calendar_feature_geometry(sae)
    assert calendar_geometry['dimensionality'] == 2

    # Test 2: Circular features (Engels et al., 2025)
    # Expected: Detect circular structure
    circularity = measure_circularity(sae.get_calendar_features())
    assert circularity > 0.8

    # Test 3: Hierarchical organization
    # Expected: Temporal features cluster together
    modularity = measure_spatial_modularity(sae.decoder_weights)
    assert modularity['temporal_lobe_score'] > 0.7

    return {
        'calendar_geometry': calendar_geometry,
        'circularity': circularity,
        'modularity': modularity,
    }
```

---

## 8. Detailed Implementation Plan

### 8.1 Phase 1: Extend SynthSAEBench (Weeks 1-3)

**Week 1: Manifold generation**

- [ ] Implement `generate_circular_manifold()`
- [ ] Implement `generate_spherical_manifold()`
- [ ] Implement `generate_toroidal_manifold()`
- [ ] Implement `embed_manifold_in_activation_space()`
- [ ] Unit tests: verify manifold properties (geodesic distances, topology)

**Week 2: Hybrid dictionary construction**

- [ ] Implement `construct_hybrid_dictionary()`
- [ ] Implement `ManifoldFeatureStats` class
- [ ] Implement `generate_activation_with_manifolds()`
- [ ] Integration tests: verify activation generation is consistent

**Week 3: Correlations and hierarchy**

- [ ] Implement `build_manifold_aware_correlation_matrix()`
- [ ] Implement `ManifoldHierarchy` class
- [ ] Implement `enforce_hierarchy_with_manifolds()`
- [ ] Generate first full dataset: 1M samples with 10 manifolds

**Milestone 1:** Generate and validate ManifoldSynthSAEBench dataset

### 8.2 Phase 2: Evaluation Metrics (Weeks 4-5)

**Week 4: Geometric metrics**

- [ ] Implement `detect_manifold_clusters()`
- [ ] Implement `analyze_decoder_subspace_geometry()`
- [ ] Implement `measure_circularity()` and `measure_sphericity()`
- [ ] Implement `manifold_alignment_score()`
- [ ] Implement `compute_geodesic_distances()`

**Week 5: Topological metrics**

- [ ] Integrate `ripser` for persistent homology
- [ ] Implement `topological_alignment()`
- [ ] Implement `curvature_estimation_accuracy()`
- [ ] Implement `ManifoldSAEBenchmark` comprehensive suite
- [ ] Validation: test metrics on toy examples with known structure

**Milestone 2:** Complete evaluation suite with validated metrics

### 8.3 Phase 3: Manifold-Aware SAE Architectures (Weeks 6-8)

**Week 6: Grouped Latent SAE**

- [ ] Implement `GroupedLatentSAE`
- [ ] Train on ManifoldSynthSAEBench
- [ ] Evaluate with new metrics
- [ ] Compare to standard SAE baseline

**Week 7: Manifold-Parametric SAE**

- [ ] Implement `CircularManifoldModule`
- [ ] Implement `SphericalManifoldModule`
- [ ] Implement `ManifoldParametricSAE`
- [ ] Train and evaluate

**Week 8: Hierarchical and Adaptive SAEs**

- [ ] Implement `HierarchicalManifoldSAE`
- [ ] Implement `AdaptiveLatentManifoldSAE`
- [ ] Comparative evaluation of all architectures
- [ ] Identify best-performing approach

**Milestone 3:** Validated manifold-aware SAE architectures

### 8.4 Phase 4: Representation Hypothesis Testing (Weeks 9-11)

**Week 9: Generate hypothesis-specific models**

- [ ] Implement pure LRH model (baseline)
- [ ] Implement manifold hypothesis model
- [ ] Implement superposition + manifolds model
- [ ] Implement hierarchical manifolds model
- [ ] Implement compositional manifolds model

**Week 10: Training dynamics experiments**

- [ ] Implement `compare_training_dynamics()`
- [ ] Track feature emergence, splitting, absorption
- [ ] Measure dead neuron rates
- [ ] Compare scaling laws across models

**Week 11: Formalize predictions**

- [ ] Run scaling experiments (vary N_latents from 1K to 32K)
- [ ] Fit power laws and measure exponents
- [ ] Count latents per ground-truth feature
- [ ] Statistical significance testing

**Milestone 4:** Completed hypothesis testing with formal predictions

### 8.5 Phase 5: Validation and LLM Comparison (Weeks 12-14)

**Week 12: Cross-model consistency**

- [ ] Implement `test_cross_model_consistency()`
- [ ] Train 5 independent models × 3 architectures = 15 SAEs
- [ ] Measure manifold consistency
- [ ] Statistical analysis

**Week 13: LLM transfer and interventions**

- [ ] Implement `test_manifold_transfer()`
- [ ] Pre-train on synthetic, fine-tune on GPT-2
- [ ] Implement `test_manifold_causality()`
- [ ] Run causal intervention experiments

**Week 14: Reproduce LLM phenomena**

- [ ] Implement `validate_against_llm_phenomena()`
- [ ] Test calendar geometry (Leask et al.)
- [ ] Test circular features (Engels et al.)
- [ ] Test hierarchical modularity (Li et al.)
- [ ] Quantitative comparison

**Milestone 5:** Validation complete, ready for publication

### 8.6 Phase 6: Scaling and Optimization (Weeks 15-16)

**Week 15: Computational optimization**

- [ ] Optimize manifold generation (vectorize, parallelize)
- [ ] Optimize evaluation metrics (batch processing)
- [ ] GPU acceleration for SAE training
- [ ] Benchmark: target 10K samples/sec generation, 1hr training per SAE

**Week 16: Large-scale experiments**

- [ ] Scale to 10M samples
- [ ] Scale to 64K latents
- [ ] Train ensemble of 50+ SAEs
- [ ] Generate comprehensive results tables and figures

**Milestone 6:** Scalable implementation ready for large-scale experiments

### 8.7 Phase 7: Paper Writing and Dissemination (Weeks 17-20)

**Week 17-18: Draft paper**

- [ ] Introduction and related work
- [ ] Methods: Manifold generation, evaluation, architectures
- [ ] Results: Tables, figures, ablations
- [ ] Discussion: Implications for LLM interpretability

**Week 19: Experiments for reviewers**

- [ ] Additional ablations based on paper narrative
- [ ] Appendix experiments
- [ ] Reproducibility: clean code, release dataset

**Week 20: Submission and release**

- [ ] Submit to ICLR/NeurIPS/ICML
- [ ] Release code on GitHub
- [ ] Release ManifoldSynthSAEBench dataset
- [ ] Blog post / Twitter thread

**Milestone 7:** Paper submitted, code and data public

---

## 9. Expected Results and Implications

### 9.1 Quantitative Predictions

Based on prior work and theoretical considerations, we predict:

| Metric                | Standard SAE on Manifolds | Manifold-Aware SAE | LRH Baseline |
| --------------------- | ------------------------- | ------------------ | ------------ |
| Independent MCC       | 0.70 ± 0.05               | 0.72 ± 0.05        | 0.75 ± 0.03  |
| Manifold Alignment    | 0.25 ± 0.10               | 0.75 ± 0.08        | N/A          |
| Topology Preservation | 0.15 ± 0.10               | 0.80 ± 0.10        | N/A          |
| Latents per Manifold  | 15 ± 5                    | 3 ± 1              | N/A          |
| Dead Neuron Rate      | 25% ± 5%                  | 12% ± 3%           | 10% ± 2%     |
| Curvature Error       | 80% ± 20%                 | 15% ± 10%          | N/A          |

### 9.2 Qualitative Insights

**Finding 1: Manifold tiling pathology**
Standard SAEs will "tile" circular manifolds with many latents, each representing a small arc. This is inefficient but locally optimal for reconstruction loss.

**Finding 2: Scaling breakdown**
Michaud et al.'s predicted pathological scaling will be empirically confirmed: in high-superposition + manifold regime, number of discovered features grows sublinearly with latent count.

**Finding 3: Architecture matters**
Manifold-parametric SAEs will achieve 3-5× better manifold alignment than standard SAEs, proving that architectural inductive biases are crucial.

**Finding 4: Hierarchy-manifold interaction**
Hierarchical manifolds will show error propagation: missing parent features → catastrophic failure to recover child manifolds.

**Finding 5: LLM correspondence**
Training dynamics on synthetic manifold models will quantitatively match known LLM SAE behaviors (feature splitting rates, dead neuron curves), validating the benchmark's realism.

### 9.3 Implications for LLM Interpretability

**Implication 1: Standard SAEs are insufficient**
If manifold features are common in LLMs (as Engels et al. suggest), current SAE architectures are fundamentally limited. The field needs manifold-aware alternatives.

**Implication 2: Evaluation must evolve**
Point-wise feature recovery (MCC, F1) misses geometric structure. Interpretability research should adopt manifold-aware evaluation.

**Implication 3: Synthetic benchmarks are crucial**
Without ground truth, we can't measure manifold recovery. SynthSAEBench-style approaches are essential for rigorous progress.

**Implication 4: Representation hypotheses are testable**
By generating synthetic models instantiating different hypotheses, we can empirically distinguish between theories of neural representation.

**Implication 5: Transfer learning potential**
If manifold-aware SAEs pre-trained on synthetic data transfer to real LLMs, this opens a new paradigm: use synthetic models to develop better interpretability tools, then apply to real models.

---

## 10. Open Questions and Future Directions

### 10.1 Open Questions

1. **Optimal manifold parameterization:** Should manifolds be discretized (as proposed) or continuous? What's the right discretization density?

2. **Manifold discovery:** Can SAEs automatically discover manifold type (circular vs. spherical vs. hyperbolic) from data, or must it be specified?

3. **Compositional manifolds:** What's the best way to represent composed manifolds (S¹ × S¹)? Factorized vs. monolithic?

4. **Manifold superposition:** Can manifolds themselves be in superposition (e.g., "day of week" and "zodiac sign" both circular, overlapping)?

5. **Scaling to high dimensions:** How do evaluation metrics scale to higher-dimensional manifolds (e.g., 10D hyperbolic spaces)?

6. **Noise vs. curvature:** How to distinguish genuine manifold curvature from noise or sampling artifacts?

### 10.2 Extensions

**Extension 1: Other manifold types**

- Hyperbolic manifolds for hierarchical features (Poincaré disk)
- Grassmann manifolds for subspace features
- Product manifolds (S¹ × ℝ × S²) for compositional structure

**Extension 2: Temporal manifolds**

- Manifolds evolving over training (do circular features emerge gradually?)
- Dynamic manifolds in recurrent networks

**Extension 3: Multi-modal manifolds**

- Vision-language aligned manifolds
- Cross-modal manifold transfer

**Extension 4: Causal manifolds**

- Interventional evaluation: steer along manifold geodesics
- Counterfactual evaluation: move off-manifold and measure impact

**Extension 5: Neuroscience connection**

- Compare to grid cells (toroidal manifolds) in entorhinal cortex
- Test if artificial manifolds resemble biological representations

### 10.3 Long-Term Vision

**Goal:** Establish manifold-aware interpretability as a standard paradigm.

**Success metrics:**

1. Manifold evaluation metrics adopted by SAE papers
2. Manifold-aware SAE architectures achieve SOTA on real LLMs
3. SynthSAEBench with manifolds becomes standard benchmark
4. Manifold hypotheses inform theoretical understanding of neural representations

**Broader impact:**

- Enable more precise steering and control of LLMs via manifold understanding
- Improve alignment research by revealing geometric structure of concepts
- Bridge machine learning and neuroscience through shared manifold framework

---

## 11. Conclusion

This research plan extends SynthSAEBench to systematically study how sparse autoencoders handle feature manifolds—a critical but understudied aspect of neural representation. By generating synthetic data with known manifold ground truth, developing manifold-aware evaluation metrics, designing architectural innovations, and testing competing representation hypotheses, we can rigorously assess when and how SAEs recover geometric structure.

The key insight is that **features are not always one-dimensional**. Temporal cycles, spatial directions, and compositional concepts naturally lie on manifolds. If interpretability tools ignore this geometry, they will fail to capture how models truly represent knowledge. By grounding our investigation in the SynthSAEBench framework—where we control ground truth and maintain scientific rigor—we can make definitive progress on understanding manifold representations in neural networks.

Our approach is not merely theoretical. We make concrete, testable predictions that can be validated against real LLM behavior. We propose practical architectures that can be deployed on real models. And we establish evaluation protocols that the community can adopt. This work has the potential to reshape how we think about feature learning, representation geometry, and the fundamental units of neural computation.

The path forward is clear: implement, evaluate, iterate, and validate. With systematic experiments on carefully designed synthetic models, we can finally answer the question: **Are manifolds the missing piece in sparse autoencoder interpretability?**

---

## References

**Key Papers on Feature Manifolds:**

- **Engels, J. E., Michaud, E. J., Liao, I., Gurnee, W., & Tegmark, M. (2025).** "Not All Language Model Features Are One-Dimensionally Linear." ICLR 2025. [https://arxiv.org/abs/2405.14860](https://arxiv.org/abs/2405.14860)

- **Li, Y., Michaud, E. J., Baek, D. D., Engels, J., Sun, X., & Tegmark, M. (2025).** "The Geometry of Concepts: Sparse Autoencoder Feature Structure." Entropy, 27(4), 344. [https://arxiv.org/abs/2410.19750](https://arxiv.org/abs/2410.19750)

- **Olah, C., & Batson, J. (2023).** "Feature Manifold Toy Model." Transformer Circuits Thread, May Update. [https://transformer-circuits.pub/2023/may-update/index.html#feature-manifolds](https://transformer-circuits.pub/2023/may-update/index.html#feature-manifolds)

- **Michaud, E. J., et al. (2024).** "Understanding Sparse Autoencoder Scaling in the Presence of Feature Manifolds." [https://arxiv.org/abs/2509.02565](https://arxiv.org/abs/2509.02565)

**Key Papers on SAEs:**

- **Bricken, T., et al. (2023).** "Towards Monosemanticity: Decomposing Language Models with Dictionary Learning." [https://transformer-circuits.pub/2023/monosemantic-features](https://transformer-circuits.pub/2023/monosemantic-features)

- **Gao, L., et al. (2025).** "Scaling and Evaluating Sparse Autoencoders." ICLR 2025. [https://cdn.openai.com/papers/sparse-autoencoders.pdf](https://cdn.openai.com/papers/sparse-autoencoders.pdf)

- **Cunningham, H., et al. (2023).** "Sparse Autoencoders Find Highly Interpretable Features in Language Models." [https://arxiv.org/abs/2309.08600](https://arxiv.org/abs/2309.08600)

**SynthSAEBench:**

- **Karvonen, A., et al. (2024).** "SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data." (See: far.ai/18572_SynthSAEBench_Evaluating.pdf)

**Geometric Representation:**

- **Park, K., Choe, Y. J., & Veitch, V. (2024).** "The Linear Representation Hypothesis and the Geometry of Large Language Models." ICML 2024.

- **Leask, P., et al. (2024).** "Calendar Feature Geometry in GPT-2 Layer 8 Residual Stream SAEs." [https://www.lesswrong.com/posts/WsPyunwpXYCM2iN6t](https://www.lesswrong.com/posts/WsPyunwpXYCM2iN6t)

---

## Appendix A: Code Structure

```
manifold-sae-bench/
├── data_generation/
│   ├── manifolds.py              # Manifold generation (circular, spherical, etc.)
│   ├── hybrid_dictionary.py      # Construct dictionary with manifolds
│   ├── activation_sampler.py     # Sample activations with manifolds
│   └── hierarchy.py               # Hierarchical manifolds
├── models/
│   ├── standard_sae.py           # Baseline L1 SAE
│   ├── grouped_latent_sae.py     # GL-SAE
│   ├── manifold_parametric_sae.py # MP-SAE
│   ├── hierarchical_sae.py       # HM-SAE
│   └── adaptive_sae.py            # ALM-SAE
├── evaluation/
│   ├── geometric_metrics.py      # Manifold alignment, geodesic preservation
│   ├── topological_metrics.py    # Persistent homology
│   ├── standard_metrics.py       # MCC, F1, variance explained
│   └── benchmark.py               # Unified evaluation suite
├── experiments/
│   ├── hypothesis_testing.py     # Generate models per hypothesis
│   ├── training_dynamics.py      # Track emergence, splitting, etc.
│   ├── scaling_laws.py            # Scaling experiments
│   └── llm_validation.py          # Compare to real LLM behaviors
├── visualization/
│   ├── manifold_plots.py         # Visualize learned manifolds
│   ├── geometry_plots.py          # PCA, tSNE of decoder weights
│   └── training_curves.py         # Loss, metrics over training
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_model_training.ipynb
    ├── 03_evaluation.ipynb
    └── 04_hypothesis_testing.ipynb
```

---

## Appendix B: Mathematical Notation Reference

| Symbol   | Meaning                                                         |
| -------- | --------------------------------------------------------------- |
| **d**    | Hidden dimension (768)                                          |
| **N**    | Number of features (16,384)                                     |
| **k**    | Intrinsic manifold dimension (1 for circle, 2 for sphere, etc.) |
| **D**    | Feature dictionary, D ∈ ℝ^{d×N}                                 |
| **x**    | Activation vector, x ∈ ℝ^d                                      |
| **c**    | Ground-truth feature coefficients, c ∈ ℝ^N                      |
| **f**    | SAE latent activations, f ∈ ℝ^{N_latents}                       |
| **x̂**   | Reconstructed activation                                        |
| **ρ_mm** | Mean max cosine similarity (superposition measure)              |
| **L0**   | Number of active features (sparsity)                            |
| **S^k**  | k-dimensional sphere                                            |
| **θ, φ** | Angular coordinates on manifolds                                |
| **Σ**    | Correlation matrix                                              |
| **F**    | Low-rank factor matrix for correlations                         |

---

**End of Research Plan**

*This document provides a comprehensive, technically rigorous framework for extending SynthSAEBench to study manifold-aware sparse autoencoders. Implementation can proceed according to the phased plan, with each milestone building toward a complete system for testing representation hypotheses and advancing interpretability research.*
