# Extending SynthSAEBench: Incorporating Feature Manifolds and Evaluating Sparse Autoencoders on Multi-Dimensional Representations

**A Comprehensive Research Proposal for Testing Manifold Representation Hypotheses in Sparse Autoencoders**

---

## 1. Introduction and Motivation

The Linear Representation Hypothesis posits that neural networks represent concepts as directions in activation space, with sparse autoencoders serving as tools to decompose activations into these interpretable features. However, recent empirical work has demonstrated that not all neural network features conform to this one-dimensional linear structure. The discovery of circular representations for periodic concepts such as days of the week and months of the year in language models suggests that some features are inherently multi-dimensional, living on manifolds embedded in the activation space rather than along single directions.

This poses a fundamental challenge for sparse autoencoder evaluation and development. The SynthSAEBench framework provides an excellent foundation for controlled experimentation with ground-truth features, incorporating realistic phenomena such as correlation, hierarchy, and superposition. However, it currently models all features as one-dimensional directions following the traditional Linear Representation Hypothesis. Extending this framework to include feature manifolds would address a critical gap, allowing us to systematically study how SAEs behave when confronted with multi-dimensional structure and to develop improved architectures that can properly decompose such representations.

The work by Michaud, Gorton, and McGrath on SAE scaling in the presence of feature manifolds reveals a potential pathology in how SAEs might handle multi-dimensional features. Their theoretical analysis suggests that when SAEs encounter feature manifolds with slowly decreasing loss curves as a function of allocated latents, the SAE may wastefully allocate many latents to "tile" common manifolds rather than discovering rarer discrete features. This phenomenon could manifest as what has been termed "interpretability dark matter," where SAEs fail to recover a substantial portion of the model's true computational structure. Understanding whether this pathology occurs in practice, and under what conditions, requires controlled experiments with ground-truth manifolds where we can precisely measure SAE behavior.

---

## 2. Technical Framework for Introducing Manifolds into SynthSAEBench

The core generative model in SynthSAEBench represents each feature as a coefficient multiplied by a unit direction vector. To incorporate manifolds, we need to extend this formulation so that some features can live on multi-dimensional subspaces rather than along single directions. The key insight is that we can represent a feature manifold as a structured subspace where the feature coefficients themselves follow geometric constraints.

### 2.1 Mathematical Formulation of Manifold Features

We extend the SynthSAEBench data generation process to support manifold features while maintaining compatibility with the existing discrete feature infrastructure. For a manifold feature indexed by $i$, instead of a scalar coefficient $c_i$ and direction vector $d_i \in \mathbb{R}^D$, we introduce a manifold embedding function $\phi_i: M_i \to \mathbb{R}^D$ where $M_i$ is an intrinsic manifold (such as a circle $S^1$, sphere $S^2$, or torus $T^2$).

The generation process for a manifold feature proceeds as follows. First, we sample a point $m_i \in M_i$ from the intrinsic manifold according to some distribution $p_{M_i}$. This point is then embedded into the activation space via the embedding function: $v_i = \phi_i(m_i)$. To incorporate the intensity or strength with which the feature fires (analogous to the magnitude of scalar coefficients for discrete features), we introduce a radial component sampled from the same rectified Gaussian distribution used for discrete features: $r_i = \text{ReLU}(\mu_i + \sigma_i \epsilon_i)$ where $\epsilon_i \sim \mathcal{N}(0,1)$.

The contribution of manifold feature $i$ to the activation is then:

$a_i = r_i \cdot \phi_i(m_i) + b_i$

$a = \sum_{i \in \text{discrete}} c_i d_i + \sum_{j \in \text{manifold}} r_j \phi_j(m_j) + b$

This formulation preserves several important properties of the SynthSAEBench framework. The firing probability $p_i$ still controls whether feature $i$ is active on a given sample. The hierarchy constraints can still be applied, with child manifold features only activating when their parent features are active. Correlation structure can be imposed through the Gaussian copula mechanism that determines which features fire together.

### 2.2 Implementing Specific Manifold Geometries

The research should begin with simple, well-understood manifold geometries that have interpretable properties and clear connections to phenomena observed in language models. I recommend implementing the following manifolds in order of increasing complexity:

#### Circular Features ($S^1$)

These represent the simplest non-trivial manifold and correspond directly to the circular day-of-week and month-of-year representations discovered in language models. For a circle in $\mathbb{R}^D$, we can parameterize the embedding as:

$\phi_{\text{circle}}(\theta) = \cos(\theta) \cdot u_1 + \sin(\theta) \cdot u_2$

where $u_1, u_2 \in \mathbb{R}^D$ are orthonormal vectors defining the plane containing the circle, and $\theta \in [0, 2\pi)$ is sampled uniformly or according to some distribution reflecting the relative frequencies of different positions on the circle.

#### Spherical Features ($S^{d-1}$)

Higher-dimensional spheres provide a way to model features that involve multiple interacting dimensions without a clear decomposition into independent lower-dimensional manifolds. For a sphere in $\mathbb{R}^D$, we sample points uniformly on $S^{d-1}$ embedded in a $(d)$-dimensional subspace of $\mathbb{R}^D$. This can be accomplished by sampling $d$ independent standard normal random variables, normalizing to unit length, and embedding into $\mathbb{R}^D$ via a $(D \times d)$ orthonormal basis matrix.

#### Toroidal Features ($T^2 = S^1 \times S^1$)

A torus represents the product of two circular features and might correspond to concepts that involve two independent periodic components. The embedding can be parameterized as:

$\phi_{\text{torus}}(\theta_1, \theta_2) = \cos(\theta_1) u_1 + \sin(\theta_1) u_2 + \cos(\theta_2) u_3 + \sin(\theta_2) u_4$

where $u_1, u_2, u_3, u_4$ are orthonormal vectors in $\mathbb{R}^D$.

#### Radial Variation

Following the findings of Michaud et al., it is critical to include variation in the radial direction (the intensity with which features fire). Their experiments demonstrated that when SAEs are trained on hollow manifolds without radial variation, they can tile the manifold with many sparsely firing latents, but when radial variation is present, SAEs tend to learn a small number of latents that approximate a basis for the manifold subspace. This suggests that radial variation fundamentally changes the SAE learning dynamics and makes the problem more realistic.

### 2.3 Controlling Manifold Complexity and Realism

To enable systematic experimentation, the manifold implementation should expose several control parameters:

**Manifold dimension:** The intrinsic dimensionality of the manifold (1 for circles, 2 for spheres and tori, etc.). This should be configurable independently of the embedding dimension.

**Embedding dimension:** The dimension of the subspace in $\mathbb{R}^D$ into which the manifold is embedded. For a $d$-dimensional manifold, this must be at least $d+1$ to embed without self-intersection for smooth manifolds.

**Radial distribution:** The parameters ($\mu_i$, $\sigma_i$) controlling the distribution of radial magnitudes. The research should explore both hollow manifolds (no radial variation, $\sigma_i = 0$) and manifolds with realistic radial variation.

**Curvature and nonlinearity:** For more advanced experiments, it would be valuable to implement manifolds with non-constant curvature or "ripples" as discussed in the Anthropic interpretability work. This could be achieved by modulating the embedding function $\phi$ with additional nonlinear components.

**Discretization:** For some experiments, it may be useful to work with discretized manifolds (e.g., sampling from a finite set of evenly spaced points on a circle) to create intermediate cases between continuous manifolds and truly discrete features.

---

## 3. Evaluation Methodology for SAEs on Manifold Features

Evaluating SAE performance on manifold features requires metrics that go beyond the standard measures used for discrete features. While metrics like Mean Correlation Coefficient (MCC) and F1 scores for binary classification remain useful, they need to be adapted and supplemented with manifold-specific evaluation criteria.

### 3.1 Manifold Reconstruction Quality

The first category of metrics assesses how well the SAE can reconstruct the geometric structure of the manifold itself, independent of whether individual latents are interpretable.

#### Subspace Alignment

For a ground-truth manifold living in a $d$-dimensional subspace $S_i \subset \mathbb{R}^D$, we want to measure whether the SAE latents associated with this manifold collectively span a similar subspace. Let $W_i$ be the matrix whose columns are the decoder directions of all SAE latents matched to manifold feature $i$. We can compute the principal angles $\theta_1, \ldots, \theta_{\min(d, \text{rank}(W_i))}$ between the true subspace $S_i$ and the learned subspace $\text{span}(W_i)$. A well-aligned reconstruction should have small principal angles. The subspace alignment score can be defined as:

$\text{Alignment}_i = \frac{1}{d} \sum_{j=1}^{d} \cos^2(\theta_j)$

with values near 1 indicating good alignment.

#### Manifold Coverage

Beyond subspace alignment, we need to verify that the SAE actually captures the full extent of the manifold rather than only a portion of it. For a manifold feature $i$, we can partition the manifold into regions and measure what fraction of regions activate at least one associated SAE latent. For circular features, this could involve dividing the circle into angular bins and computing the coverage fraction. Low coverage would indicate that the SAE has only learned to represent part of the manifold's structure.

#### Geometric Distortion

Even if the SAE captures the correct subspace and full coverage, it might distort the intrinsic geometry of the manifold. For instance, an SAE might learn to represent a circle as an elongated ellipse or irregular closed curve. To measure this, we can compare distances between points on the true manifold $M_i$ to distances between their reconstructions in the learned representation. Specifically, for sampled points $m^{(1)}, \ldots, m^{(n)}$ from the manifold and their SAE reconstructions $\hat{m}^{(1)}, \ldots, \hat{m}^{(n)}$, we compute:

$\text{Distortion} = \frac{1}{n(n-1)} \sum_{j \neq k} \left| \frac{d_M(m^{(j)}, m^{(k)})}{d_{\mathbb{R}^D}(\hat{m}^{(j)}, \hat{m}^{(k)})} - 1 \right|$

where $d_M$ is the geodesic distance on the manifold and $d_{\mathbb{R}^D}$ is Euclidean distance in activation space.

### 3.2 Latent Allocation and Tiling Analysis

A key concern raised by the theoretical work on manifold scaling is whether SAEs wastefully allocate many latents to tile manifolds. This requires metrics that characterize how latents are distributed across features.

#### Latents per Feature

For each ground-truth feature (both discrete and manifold), we count how many SAE latents are primarily associated with it. This is determined by the matching procedure used for MCC calculation. The distribution of latents per feature reveals whether certain features (particularly common manifold features) are capturing a disproportionate share of SAE capacity. We should pay particular attention to manifold features and compute:

$\text{Tiling Factor}_i = \frac{\text{number of latents matched to feature } i}{\text{intrinsic dimension of feature } i}$

For discrete features, the intrinsic dimension is 1, so a tiling factor near 1 is ideal. For manifold features, the tiling factor tells us how many times the manifold has been "oversampled" relative to what would be needed for a minimal basis representation.

#### Sparsity Within Manifolds

When multiple latents tile a manifold, we expect them to fire sparsely relative to each other (each covering a portion of the manifold). We can quantify this by examining the overlap in activation patterns. For latents $j_1, j_2$ both matched to manifold feature $i$, compute the fraction of samples where both latents are simultaneously active:

$\text{Overlap}(j_1, j_2) = \frac{|\{x : \hat{f}_{j_1}(x) > 0 \text{ and } \hat{f}_{j_2}(x) > 0\}|}{|\{x : \text{feature } i \text{ active on } x\}|}$

Low overlap indicates that latents are indeed tiling different regions of the manifold.

#### Latent Clustering Analysis

For manifolds embedded in subspaces, latents tiling the manifold should cluster in decoder direction space. We can apply clustering algorithms to the set of decoder vectors matched to a manifold feature and measure cluster coherence using silhouette scores or similar metrics. Additionally, we can visualize the decoder directions projected onto the ground-truth manifold subspace to understand the spatial distribution of learned representations.

### 3.3 Functional Decomposition Quality

Beyond geometric properties, we need to assess whether the SAE's representation of manifolds supports the kind of interpretable decomposition that motivates SAE research in the first place.

#### Axis-Aligned Decomposition Score

For some manifolds, there exists a natural basis (e.g., $\sin$ and $\cos$ components for circles) that provides maximal interpretability. We can measure whether the SAE discovers such decompositions by comparing learned latents to these canonical bases. For a circle parameterized by $\theta$, the ideal decomposition uses latents corresponding to $\cos(\theta)$ and $\sin(\theta)$. We can compute the correlation between actual latent activations and these ground-truth components:

$\text{Basis Quality} = \max_{\text{permutation}} \frac{1}{d} \sum_{k=1}^{d} |\text{corr}(\hat{f}_{j_k}, \phi_k)|$

where $\phi_k$ are the ground-truth basis components and the max is taken over permutations and sign flips of the learned latents.

#### Compositional Coding

Some manifolds might be better represented compositionally. For instance, a torus might be decomposed into two independent circular features. We should measure whether SAEs discover such factorizations by testing whether latent activations for a product manifold can be written as products of latent activations for the component manifolds.

### 3.4 Probing and Classification Performance

The probing framework from SynthSAEBench can be extended to manifold features, though the notion of "classification" needs to be generalized.

#### Region Classification

Instead of binary classification (feature active vs. inactive), we can define classification tasks based on which region of the manifold is active. For circular features, this could involve classifying which angular quadrant is active. For each region, we train a linear probe on SAE latent activations and measure classification accuracy.

#### Manifold Position Regression

A more fine-grained evaluation uses regression instead of classification. Given SAE latent activations, can we accurately predict the position on the manifold? For instance, for a circular feature, can we predict the angle $\theta$ from the latent activations? This can be measured using mean squared error between predicted and true manifold coordinates.

#### Disentanglement Metrics

For features represented as manifolds, we want SAE latents to disentangle the manifold direction from radial intensity. We can adapt standard disentanglement metrics from the representation learning literature, such as computing mutual information between latent activations and manifold coordinates versus radial magnitudes.

---

## 4. Research Roadmap for Testing Manifold Representation Hypotheses

The central research question is whether observed phenomena in LLM SAEs can be explained by the presence of feature manifolds, and conversely, whether synthetic models with manifolds exhibit similar behaviors to real SAEs trained on LLMs. This requires a systematic experimental program that progressively increases realism.

### 4.1 Phase 1: Isolated Manifold Experiments

The research should begin with the simplest possible setup: synthetic data models containing a single feature manifold and no other features. This eliminates confounds from feature interactions and allows us to characterize the fundamental SAE learning dynamics on manifolds.

#### Single Circle Experiments

Create synthetic datasets where each activation is a point sampled from a circle embedded in $\mathbb{R}^D$ with added noise. Systematically vary the embedding dimension $D$, the amount of radial variation in firing magnitudes, the amount of Gaussian noise added to activations, and the radius of the circle. Train SAEs with different architectures (standard L1, TopK, Matching Pursuit, Matryoshka) and varying numbers of latents. This recreates and extends the experiments in Michaud et al.'s work, allowing us to carefully characterize the loss scaling curves $L(n)$ for different manifold types and SAE architectures. 

Key measurements include: the number of latents used to represent the circle, whether latents form a basis (2 latents for a circle) or tile the manifold, the reconstruction loss as a function of number of latents, and the geometric quality of the learned representation.

#### Varying Manifold Geometry

Extend these experiments to different manifold types (spheres of varying dimension, tori, more complex shapes) while keeping each dataset to a single manifold. This establishes how manifold geometry affects SAE behavior. We expect that hollow manifolds (no radial variation) will be more susceptible to tiling, while manifolds with radial variation should encourage basis-like solutions. The dimensionality of the manifold should affect how many latents are needed for good reconstruction.

#### Comparing to Theoretical Predictions

The experiments in this phase should be designed to validate or refute the theoretical predictions from Michaud et al. We can measure the empirical loss scaling exponent $\beta$ from the $L(n) \propto n^{-\beta}$ relationship and compare it to theoretical predictions based on manifold geometry. We should also verify whether the predicted pathological scaling behavior (where $\beta < \alpha$ leads to over-allocation of latents to common manifolds) actually occurs in practice.

### 4.2 Phase 2: Manifolds with Discrete Features

The next phase introduces the core tension: SAEs must balance allocating latents to reconstruct manifolds versus discovering discrete features. This directly tests whether manifold tiling crowds out rare feature discovery.

#### Zipfian Feature Distribution with One Manifold

Create a SynthSAEBench-style dataset with many discrete features following a Zipfian frequency distribution, but replace the most frequent discrete feature with a manifold feature. This manifold feature should fire with the highest probability $p_{\max}$ while all other features are discrete. Train SAEs of varying widths and measure: the number of latents allocated to the manifold versus discrete features, the number of discrete features discovered as a function of total SAE width, whether this differs from the all-discrete baseline, and the relationship between manifold loss reduction and discrete feature discovery.

The key question is whether the SAE exhibits the pathological scaling predicted when $\beta < \alpha$. According to the theory, if the manifold has a slowly decreasing loss curve ($L(n) \propto n^{-\beta}$ with small $\beta$) and the discrete feature frequencies decay as $p_i \propto i^{-(1+\alpha)}$ with $\alpha > \beta$, then most latents should be allocated to tiling the manifold rather than discovering rare discrete features. We can test this by plotting the number of discovered discrete features $D(N)$ versus total SAE width $N$ and comparing the slope to the theoretical prediction $D(N) \propto N^{(1+\beta)/(1+\alpha)}$.

#### Multiple Manifolds with Hierarchy

Introduce multiple manifold features at different levels of the feature hierarchy. For instance, a parent feature might be a discrete feature representing "temporal concepts" with child features being circular manifolds for "days of week" and "months of year." This tests whether hierarchical relationships between manifolds and discrete features are learned correctly and whether the manifold tiling problem is exacerbated or mitigated by hierarchy.

#### Correlated Manifolds

Implement scenarios where multiple manifold features are correlated, meaning they tend to fire together. For example, two circular features representing related periodic phenomena might be coupled through the correlation matrix. This tests whether SAEs can disentangle correlated manifolds or whether they conflate them into a single higher-dimensional representation.

### 4.3 Phase 3: Realistic Synthetic Benchmarks with Manifolds

Building on Phase 2, we create realistic synthetic benchmarks that incorporate manifolds alongside all other SynthSAEBench phenomena: correlation, hierarchy, superposition, and realistic firing distributions.

#### SynthSAEBench-16k with Manifolds

Extend the standard SynthSAEBench-16k benchmark by converting a subset of features to manifolds. Start conservatively by converting perhaps 1-5% of features to manifolds (approximately 164-820 features). These should be distributed across the frequency spectrum, with some high-frequency manifolds and some rare manifolds. The manifolds should have varying intrinsic dimensions (mostly circles with some higher-dimensional spheres or tori) and should respect the existing hierarchy structure (a manifold can have discrete or manifold children).

For this benchmark, we need to specify clear ground-truth properties:

**Manifold catalog:** A table listing each manifold feature, its intrinsic dimension, embedding dimension, position in the hierarchy, firing probability, and radial distribution parameters.

**Expected behavior:** Theoretical predictions for how SAEs should allocate latents based on the manifold loss curves and feature frequencies.

**Comparison baselines:** Results from the original all-discrete SynthSAEBench-16k to understand the delta introduced by manifolds.

#### Parameterized Manifold Benchmarks

Create a family of benchmark variants where the proportion of manifold features, their dimensionality distribution, and their position in the frequency ordering can be varied systematically. This allows us to understand how robust SAE architectures are to different levels of manifold prevalence. For instance, one might create benchmarks with 1%, 5%, 10%, and 25% manifold features to see at what point manifolds begin to seriously degrade SAE performance.

### 4.4 Phase 4: Comparing Synthetic and LLM Behavior

The ultimate goal is to understand whether the behaviors observed in synthetic benchmarks with manifolds correspond to phenomena seen in SAEs trained on real language models. This requires careful comparison experiments.

#### Matched Phenomena Identification

First, identify specific phenomena in LLM SAEs that might be explained by manifolds. The circular day-of-week and month-of-year features provide clear examples. Other candidates might include: latent clusters with very high nearest-neighbor cosine similarity (suggesting tiling of a manifold), latents that fire on semantically related but geometrically structured concepts (e.g., temperature-related terms that might live on a continuous spectrum), and cases where many latents seem to represent the same concept with slightly different activation patterns.

#### Synthetic Recreation

For each identified phenomenon, create a targeted synthetic dataset that instantiates the hypothesized manifold structure. For the days-of-week example, create a circular manifold feature with 7 approximately evenly-spaced concentration points corresponding to the seven days. Train SAEs on this synthetic data and compare the learned representations to those found in LLMs. 

Key comparisons include: the number of latents allocated to the feature, the geometric arrangement of decoder directions, the activation patterns (do latents fire on disjoint vs. overlapping regions?), and intervention experiments where we ablate or modify latents and measure impact on task performance.

#### Scaling Law Comparisons

A particularly powerful comparison involves scaling laws. Train SAEs of varying widths on both synthetic manifold benchmarks and on LLM activations, measuring the same metrics (reconstruction loss, feature discovery rate, latent allocation patterns) for both. If manifolds are a significant factor in LLM representations, we should observe similar scaling behavior. 

Specifically, look for evidence of the $\beta < \alpha$ pathological regime in both settings: sublinear discovery rate $D(N) \propto N^\gamma$ with $\gamma < 1$, reconstruction loss scaling $L(N) \propto N^{-\beta}$ with exponent matching the manifold geometry rather than the feature frequency distribution, accumulation of latents on high-frequency features evidenced by very uneven latent-per-feature     distributions.

#### Geometry Analysis

Extract geometric statistics from both synthetic and real SAE latents. For instance, compute the distribution of nearest-neighbor cosine similarities for decoder directions, perform clustering analysis to identify groups of related latents, measure the effective dimensionality of latent decoder spaces using principal component analysis or participation ratio metrics. If manifolds are prevalent in LLMs, we should see statistical signatures in real SAEs that match those from synthetic experiments with manifolds.

### 4.5 Phase 5: Developing Manifold-Aware SAE Architectures

Once we understand how existing SAE architectures handle manifolds, we can design improved architectures specifically optimized for mixed discrete-manifold representations.

#### Explicit Manifold Decomposition

One approach is to modify the SAE architecture to explicitly separate discrete and manifold representations. The encoder could have two pathways: a standard sparse encoder for discrete features and a separate module that predicts manifold coordinates. The decoder would then reconstruct as:

$\hat{a} = \sum_j w_j^{\text{discrete}} \hat{f}_j^{\text{discrete}} + \sum_k \phi_k(\hat{m}_k^{\text{manifold}})$

where $\hat{m}_k$ are predicted manifold positions. This architecture would need to learn which features should be represented discretely versus as manifolds, potentially using a gating mechanism or end-to-end learning with appropriate inductive biases.

#### Regularization for Basis Solutions

If tiling is indeed problematic, we can introduce regularizations that explicitly encourage SAEs to learn basis-like representations for manifolds. For instance, penalize having too many latents with very high cosine similarity, or introduce a manifold-aware sparsity penalty that encourages latents matched to the same feature to be orthogonal in their decoder directions. The challenge is distinguishing between beneficial redundancy (multiple latents covering different regions of a manifold) and wasteful tiling.

#### Hybrid Dictionary Learning

Combine ideas from classical dictionary learning (where basis functions can have arbitrary shapes) with the sparse coding objective of SAEs. This might involve allowing some SAE decoder directions to be nonlinear functions rather than pure linear directions, effectively learning parameterized manifold embeddings. The sparse code would then include both standard linear coefficients and manifold position parameters.

---

## 5. Validation Against LLM Phenomena

To ensure that synthetic experiments with manifolds are addressing real issues rather than artificial ones, we need constant feedback from LLM observations.

### 5.1 Diagnostic Signals from Real SAEs

Several observable properties of SAEs trained on LLMs can serve as diagnostic signals for manifold prevalence:

#### Latent Redundancy

If many latents represent the same semantic concept with only slight variations in their activation patterns, this suggests manifold tiling. The synthetic experiments should reproduce this signature when manifolds are present, particularly when radial variation is limited.

#### Anomalous Scaling

If we observe that increasing SAE width does not proportionally increase the number of discovered interpretable features (measured through auto-interp or manual inspection), this is consistent with the pathological manifold scaling where latents accumulate on common manifolds. The synthetic benchmarks should exhibit similar sublinear discovery rates when manifolds with small $\beta$ dominate.

#### Geometric Clustering

Clusters of latents with very similar decoder directions (high cosine similarity) but different activation sparsity suggest manifold decomposition. The synthetic experiments should identify conditions under which such clusters emerge and characterize their properties.

#### Intervention Inconsistencies

If ablating multiple SAE latents has similar effects on model behavior (suggesting they represent the same underlying feature), this could indicate manifold tiling. The synthetic framework allows us to test this by comparing intervention effects on ground-truth manifold features versus truly distinct discrete features.

### 5.2 Specific Test Cases

#### Circular Features

The days-of-week and months-of-year examples provide excellent test cases because their circular structure is well-established. Create synthetic circular features with similar properties (7-point and 12-point concentration distributions on circles) and compare SAE behavior in detail. Do both synthetic and real SAEs learn approximately 2 basis-like latents (one sine, one cosine)? Do they tile the circle with many sparse latents? How does the number of latents depend on SAE width?

#### Numerical Magnitude

Representations of numerical magnitude might live on one-dimensional manifolds (number lines). Create synthetic datasets with features representing continuous quantities and test whether SAEs can learn representations that capture ordinality and approximate magnitude while remaining interpretable.

#### Compositional Concepts

Some concepts might naturally live on product manifolds (e.g., temperature and humidity forming a 2D space for weather description). The synthetic framework can explicitly construct such product manifolds and test whether SAEs discover factorized representations.

---

## 6. Experimental Protocol and Best Practices

To ensure reproducibility and clear interpretation of results, the research should follow rigorous experimental protocols.

### 6.1 Controlled Comparisons

Every experiment with manifolds should include a matched baseline with discrete features. This means creating pairs of synthetic datasets: one where feature $i$ is a manifold, and one where feature $i$ is a discrete feature with the same firing probability and magnitude distribution. Comparing SAE behavior across these pairs isolates the effect of manifold structure.

### 6.2 Multiple Seeds and Statistical Testing

SAE training has stochasticity from initialization and from the stochastic gradient descent process. All experiments should use at least 5 random seeds, with statistical comparisons (e.g., t-tests or non-parametric tests) used to assess whether differences between conditions are significant. The variance across seeds should be reported (e.g., as shaded regions in plots) to help interpret results.

### 6.3 Hyperparameter Sensitivity

The behavior of SAEs on manifolds will likely depend on hyperparameters such as the sparsity coefficient $\lambda$, learning rate, and architecture-specific settings. Systematic hyperparameter sweeps should be conducted, particularly for critical comparisons between SAE architectures. Document which hyperparameter settings were found to work best and whether manifolds require different hyperparameter regimes than discrete features.

### 6.4 Scaling Experiments

Both computational scaling (larger models, more training steps) and data scaling (more features, higher dimensionality) should be explored. The manifold scaling theory makes specific predictions about asymptotic behavior as the number of latents $N \to \infty$, so we need experiments at multiple scales to test whether predicted scaling regimes are reached in practice.

---

## 7. Expected Outcomes and Interpretation

This research program would yield several valuable outcomes:

### Empirical Understanding

We would gain a clear empirical understanding of how different SAE architectures handle manifold features under controlled conditions. This includes identifying which architectures are most prone to manifold tiling, quantifying the computational cost of representing manifolds, and determining what geometric properties of manifolds most affect SAE learning.

### Theoretical Validation

The synthetic experiments would test the theoretical predictions from the manifold scaling model, either validating the pathological scaling concern or identifying conditions under which it does not materialize in practice. This could refine our understanding of when manifolds pose serious interpretability challenges.

### Architectural Insights

By comparing standard and novel SAE architectures on synthetic manifold benchmarks, we can identify architectural features that improve manifold handling. This might lead to new SAE variants specifically designed for representations with geometric structure.

### Diagnostic Tools

The metrics and evaluation methodologies developed for synthetic manifolds could be adapted to real SAEs, providing diagnostic tools for identifying manifold structure in LLM representations without access to ground truth.

### Benchmark Standards

Well-designed synthetic manifold benchmarks could become standard evaluation tools for SAE research, analogous to how SynthSAEBench provides a controlled testbed for studying discrete features. This would enable fair comparisons between proposed SAE improvements.

---

## 8. Conclusion

The ultimate goal is to determine whether feature manifolds are a significant factor limiting SAE interpretability in real neural networks, and if so, to develop improved methods for decomposing representations that include both discrete features and multi-dimensional geometric structure. This research would bridge the gap between theoretical concerns about manifold scaling and practical SAE development, grounded in realistic synthetic experiments that can be systematically compared to LLM phenomena.

The proposed research program provides a systematic path from simple isolated manifold experiments through increasingly realistic synthetic benchmarks to direct comparisons with LLM SAEs. By following the SynthSAEBench philosophy of controlled experimentation with ground truth while extending it to handle multi-dimensional features, we can rigorously test the manifold hypothesis and develop improved sparse autoencoder architectures that handle the full complexity of neural network representations.

---

## Key References

**SynthSAEBench Paper:** Chanin, David. "SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data"

**Manifold Scaling Theory:** Michaud, Eric J.; Gorton, Liv; McGrath, Tom. "Understanding sparse autoencoder scaling in the presence of feature manifolds"

**Multi-Dimensional Features in LLMs:** Engels, Joshua; Michaud, Eric J; Liao, Isaac; Gurnee, Wes; Tegmark, Max. "Not All Language Model Features Are One-Dimensionally Linear"

**Sparse Manifold Transform:** Chen, Yubei; Paiton, Dylan M.; Olshausen, Bruno A. "The Sparse Manifold Transform"

**Representation Manifolds:** Modell, Alexander; Rubin-Delanchy, Patrick; Whiteley, Nick. "The Origins of Representation Manifolds in Large Language Models"
