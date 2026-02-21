# Integrated Research Implementation Plan: Synthetic Benchmarking for Universal Feature Geometry

## Research Document: Detailed Implementation of Joint Research Directions

**Authors:** Research Team  
**Date:** February 2026  
**Version:** 1.0

---

## 1. Complementary Research Questions

### Implementation Overview

This research direction focuses on creating a unified experimental framework that bridges the gap between controlled synthetic evaluation and real-world cross-model feature comparison. The core implementation involves developing a "synthetic model zoo" where we can systematically vary properties that affect feature universality.

### Detailed Implementation Steps

**Task 1.1: Multi-Model Synthetic Generator Development**

Create an extension to the SynthSAEBench codebase that generates pairs or families of related synthetic models with controlled differences. Each synthetic model pair will share a base feature set but diverge in specific, measurable ways:

- Implement a `SyntheticModelFamily` class that takes a base configuration (number of features N, hidden dimension D, superposition level ρₘₘ) and generates K variants
- Add parameters controlling "model divergence": `feature_overlap_ratio` (0.5-1.0, determining what fraction of features are shared), `basis_rotation_angle` (0-90 degrees, controlling how much the feature basis rotates between models), and `superposition_delta` (±0.1, allowing different models to have different superposition levels)
- Implement a ground-truth correspondence matrix C ∈ ℝ^(N₁×N₂) where C_ij = 1 if feature i in Model 1 corresponds to feature j in Model 2, enabling perfect evaluation of feature matching algorithms

**Task 1.2: Controlled Divergence Experiments**

Design experiments that systematically vary one aspect of model difference while holding others constant:

- **Experiment 1A: Basis Rotation Only** - Generate model pairs that have identical features but rotated bases (angles: 0°, 15°, 30°, 45°, 60°, 75°, 90°). Train SAEs on each and measure whether SVCCA/RSA can recover similarity despite basis differences.
- **Experiment 1B: Partial Feature Overlap** - Generate models where 50%, 60%, 70%, 80%, 90%, 100% of features are shared. This tests whether universality measures degrade gracefully as models become more different.
- **Experiment 1C: Different Superposition Regimes** - Create model pairs where Model 1 has ρₘₘ=0.10 and Model 2 ranges from ρₘₘ=0.10 to ρₘₘ=0.30, testing how superposition differences affect feature correspondence.

**Task 1.3: Validation Metrics Framework**

Implement comprehensive metrics that leverage ground truth:

- **Ground-Truth MCC (GT-MCC)**: Instead of using Hungarian matching on learned features, use the known correspondence matrix C to compute the "oracle MCC" - the best possible MCC if features were perfectly matched
- **Transformation Quality Score (TQS)**: After learning a transformation T from SAE₁ to SAE₂, measure TQS = correlation(T(W₁), W₂_corresponding) using ground truth correspondences
- **Universality Gap Metric**: Define UG = GT-MCC - Empirical-MCC, quantifying how much feature universality is lost due to SAE limitations vs. being genuinely absent in the models

**Task 1.4: Integration with Real Model Benchmarking**

Create a pipeline that uses synthetic experiments to predict real-world behavior:

- Train SAEs with various architectures on synthetic model pairs across all divergence conditions
- Build a regression model predicting real-world SVCCA scores from: (1) synthetic GT-MCC, (2) synthetic TQS, (3) SAE architecture type, (4) model size ratio
- Validate predictions on held-out real model pairs (e.g., Gemma-2-9B vs Gemma-2-27B)

**Expected Outcomes:**

- A flexible synthetic model generator supporting 50+ divergence configurations
- Quantitative understanding of how each type of model difference affects feature universality
- Predictive models for real-world universality with R²>0.7

---

## 2. SynthSAEBench Results Supporting Weak Universality

### Implementation Overview

This research direction focuses on characterizing the "weak universality regime" - the observation that SAEs capture features analogously but not identically. We implement experiments that quantify the spectrum from strong to weak universality.

### Detailed Implementation Steps

**Task 2.1: Architecture Comparison Framework**

Systematically compare all major SAE architectures on identical synthetic data to understand their different "views" of the same feature space:

- Use SynthSAEBench-16k as the standard testbed
- Train 5 seeds each of: Standard L1, TopK, BatchTopK, Matryoshka, JumpReLU, Gated, Matching Pursuit
- For each architecture pair (e.g., Matryoshka vs. Matching Pursuit), compute:
  - **Architecture Agreement Score**: Percentage of features where both architectures identify the same ground-truth feature (using GT correspondences)
  - **Complementary Coverage**: Percentage of ground-truth features captured by at least one architecture but not both
  - **Cross-Architecture SVCCA**: Apply SVCCA between feature spaces learned by different architectures on the same data

**Task 2.2: Reconstruction-Interpretability Trade-off Surface**

Map the Pareto frontier between reconstruction quality and feature interpretability:

- For each architecture and hyperparameter configuration, plot points in (Reconstruction MSE, MCC, F1-score) space
- Fit a Pareto frontier surface to identify non-dominated configurations
- Identify "reconstruction specialists" (high MSE, low MCC - like Matching Pursuit) and "interpretability specialists" (low MSE, high MCC - like Matryoshka)
- Implement ensemble methods that combine specialists: train both types of SAEs, use reconstruction specialists for generation tasks and interpretability specialists for feature analysis

**Task 2.3: Weak Universality Quantification**

Develop metrics that specifically measure "analogous but not identical" feature similarity:

- **Angular Similarity with Tolerance (AST)**: Instead of requiring features to be exactly aligned, measure the percentage of feature pairs within θ degrees of alignment (test θ = 5°, 10°, 15°, 20°)
- **Functional Equivalence Score (FES)**: For paired features f₁ and f₂, compute FES = correlation(f₁·dataset, f₂·dataset) - do they activate on the same tokens even if their weight vectors differ?
- **Semantic Preservation Score (SPS)**: For semantically labeled features (using the concept categories from the universality paper), measure what percentage maintain their semantic label across architectures

**Task 2.4: Stability Analysis Across Training**

Track how weak universality emerges during SAE training:

- Save SAE checkpoints every 10M tokens during training
- At each checkpoint, compute universality metrics (SVCCA, MCC) between checkpoints from different architecture trainings
- Identify whether architectures start similar and diverge, or start different and converge
- Test hypothesis: "Weak universality emerges early (first 50M tokens) and remains stable, while strong universality (exact feature matching) never fully develops"

**Task 2.5: Controlled Analogy Experiments**

Create synthetic scenarios where we know features should be analogous but not identical:

- Generate Model A with features [f₁, f₂, f₃] and Model B with features [f₁, 0.9·f₂ + 0.1·f₃, 0.1·f₂ + 0.9·f₃] (controlled feature mixing)
- Train SAEs on both models and verify that:
  - Strong matching (exact feature recovery) succeeds for f₁
  - Weak matching (SVCCA) succeeds for the mixed features
  - Functional equivalence is preserved for all features
- Use this as a calibration for real-world experiments: if methods work on controlled analogies, they should work on natural analogies

**Expected Outcomes:**

- Quantitative spectrum from strong to weak universality (0.0 = no similarity to 1.0 = identical features)
- Understanding of which architectures are "natural analogues" (high SVCCA despite different features)
- Ensemble methods achieving MCC>0.75 and MSE<0.05 simultaneously by combining specialists

---

## 3. Superposition as a Source of Non-Identical Features

### Implementation Overview

This research direction investigates how superposition creates a fundamental ambiguity in feature decomposition, leading to multiple valid but non-identical solutions. We implement experiments that directly manipulate superposition levels and measure the resulting decomposition diversity.

### Detailed Implementation Steps

**Task 3.1: Superposition Spectrum Experiments**

Extend SynthSAEBench to generate models across a fine-grained superposition spectrum:

- Create 20 synthetic models with ρₘₘ ranging from 0.05 to 0.40 in increments of ~0.018
- For each model, train 3 seeds each of 5 SAE architectures (Standard L1, TopK, Matryoshka, JumpReLU, Matching Pursuit)
- For each (superposition_level, architecture) pair, compute:
  - **Decomposition Diversity (DD)**: Standard deviation of MCC scores across the 3 seeds - high DD indicates superposition creates multiple valid solutions
  - **Architecture-Specific Response (ASR)**: Compare how different architectures respond to the same superposition level - do some architectures handle high superposition better?

**Task 3.2: Mechanistic Analysis of MP-SAE Overfitting**

Implement detailed analysis of the discovered phenomenon where Matching Pursuit SAEs overfit superposition noise:

- For models with high superposition (ρₘₘ > 0.2), train MP-SAEs and Standard L1 SAEs
- Extract specific examples where:
  - MP-SAE achieves lower reconstruction error
  - But has lower MCC (worse ground-truth feature recovery)
- Visualize the weight vectors: show that MP-SAE splits one ground-truth feature into multiple SAE features that collectively reconstruct better but individually correspond worse
- Implement a "superposition exploitation score": Measure the average number of SAE features needed to reconstruct each ground-truth feature - MP-SAEs should have higher scores

**Task 3.3: Optimal Decomposition Under Superposition**

Formulate and solve the theoretical problem of optimal feature decomposition under superposition:

- Given: Ground-truth features D with known superposition ρₘₘ
- Find: SAE decoder W that optimizes a weighted combination of reconstruction and feature recovery: L = α·MSE + (1-α)·(1-MCC)
- Solve this optimization problem for various α values (0.0, 0.25, 0.5, 0.75, 1.0) and compare solutions to actual SAE architectures
- Hypothesis: Different SAE architectures implicitly optimize different α values (MP-SAEs ≈ α=1.0, Matryoshka ≈ α=0.3)

**Task 3.4: Superposition-Aware Feature Matching**

Develop matching algorithms that explicitly account for superposition:

- Traditional matching: One-to-one Hungarian algorithm
- Superposition-aware matching: Allow many-to-many correspondences with weights
- Implement "Soft Correspondence Matrix" S where S_ij ∈ [0,1] represents the strength of correspondence between SAE feature i and ground-truth feature j
- Use this to compute "Superposition-Adjusted MCC" (SA-MCC) that credits partial correspondences
- Test whether SA-MCC better correlates with functional feature quality than standard MCC

**Task 3.5: Cross-Model Superposition Transfer**

Test whether SAEs trained on high-superposition models transfer better to other high-superposition models:

- Create three model types: Low Superposition (LS, ρₘₘ=0.08), Medium Superposition (MS, ρₘₘ=0.15), High Superposition (HS, ρₘₘ=0.25)
- Train SAEs on LS₁ and test transferability to LS₂, MS₁, HS₁
- Train SAEs on HS₁ and test transferability to HS₂, MS₁, LS₁
- Hypothesis: "HS→HS transfer succeeds better than LS→HS transfer" (SAEs learn superposition-handling strategies that transfer within regime)

**Task 3.6: Validation on Real LLMs**

Estimate superposition levels in real LLMs and test predictions:

- Use the eigenvalue decay of activation covariance matrices as a proxy for superposition (fast decay = low superposition)
- Classify real LLM layers into estimated superposition regimes
- Test whether SAE transferability between real models correlates with similarity of estimated superposition levels
- Example: Pythia-70m layer 3 and Pythia-160m layer 5 should have similar superposition if they show high universality in the original paper

**Expected Outcomes:**

- Quantitative relationship: "Each 0.05 increase in ρₘₘ reduces expected MCC by ~0.08 and increases decomposition diversity by ~0.12"
- Superposition-aware matching algorithms improving effective MCC by 15-20% in high-superposition regimes
- Validation that real-world universality patterns match synthetic superposition predictions

---

## 4. Hierarchy and Correlation Create Analogous Features

### Implementation Overview

This research direction explores how hierarchical feature relationships and correlation patterns create structured feature spaces that are similar across models even when individual features differ. We implement experiments that independently manipulate hierarchy and correlation to measure their effects on analogous universality.

### Detailed Implementation Steps

**Task 4.1: Hierarchical Structure Experiments**

Systematically vary hierarchical properties in synthetic models:

- **Hierarchy Depth Sweep**: Create models with hierarchy depths 0 (flat), 1, 2, 3, 4, 5 levels while keeping total features constant (N=16384)
- **Branching Factor Sweep**: For fixed depth=3, vary branching factors from 2 (binary trees) to 8 (octary trees)
- **Mutual Exclusivity Variations**: Test 0%, 50%, 100% mutual exclusivity among sibling features
- For each configuration:
  - Train SAEs from multiple architectures
  - Measure whether hierarchical structure is preserved: compute "Hierarchy Recovery Score (HRS)" = percentage of parent-child relationships in ground truth that are also found in nearest-neighbor relationships in SAE feature space
  - Test cross-model transfer: Do SAEs trained on depth-3 models transfer better to other depth-3 models than to depth-5 models?

**Task 4.2: Semantic Subspace Analysis**

Implement detailed analysis of the semantic subspace phenomenon discovered in the universality paper:

- Use the 8 concept categories from the universality paper (Time, Calendar, Nature, People/Roles, Emotions, MonthNames, Countries, Biology)
- For each category in synthetic data:
  - Assign ground-truth features to categories (e.g., 200 "time" features including hierarchical relationships like "decade→year→month→day")
  - Train SAEs and identify which SAE features correspond to each category
  - Measure **Subspace Coherence Score (SCS)**: Are category features clustered in SAE space? Compute average pairwise cosine similarity within vs. between categories
- Generate model pairs with identical category structures but different individual features
- Test hypothesis: "Category subspaces show higher SVCCA similarity than the overall feature space"

**Task 4.3: Correlation Pattern Experiments**

Use the low-rank correlation structure from SynthSAEBench to study correlation effects:

- Generate models with correlation rank r ∈ {10, 25, 50, 100, 200} and correlation scale s ∈ {0.025, 0.050, 0.075, 0.100, 0.150}
- For each (r, s) configuration:
  - Train SAEs and measure "Correlation Recovery" = percentage of highly correlated feature pairs (correlation > 0.5) in ground truth that are also correlated in SAE feature activation space
  - Measure how correlation affects feature matching: Do highly correlated features in Model 1 map to highly correlated features in Model 2?
- Create "correlation transfer tasks": Train on Model 1, use learned correlations to improve feature matching in Model 2

**Task 4.4: Hierarchical and Correlation Interaction Studies**

Test whether hierarchy and correlation have synergistic or antagonistic effects:

- Create a 2D sweep: Hierarchy depth {0, 2, 4} × Correlation strength {None, Low, High}
- Measure how each factor independently and jointly affects:
  - Feature recovery (MCC, F1)
  - Cross-model universality (SVCCA, RSA)
  - Semantic subspace preservation
- Hypothesis: "Hierarchy and correlation both improve universality, and their effects are super-additive (combined effect > sum of individual effects)"

**Task 4.5: Transfer Learning Using Structure**

Implement structure-aware transfer learning methods:

- **Hierarchy-Guided Matching**: When matching features between models, prefer matches that preserve parent-child relationships (if feature A matches feature X, and A's children are B and C, prioritize matching B and C to X's children)
- **Correlation-Aware Transformation**: Learn transformation matrices T that preserve correlation structure: minimize ||corr(W₁) - corr(T(W₂))||
- **Semantic Subspace Alignment**: For each semantic category, learn a separate transformation and blend them for overall feature matching
- Compare structure-aware methods to structure-agnostic baselines

**Task 4.6: Real-World Semantic Validation**

Apply semantic subspace analysis to real LLMs:

- Use the exact semantic categories from the universality paper's experiments
- For model pairs shown to have high universality (e.g., Pythia-70m/160m layers 2-3 vs 4-7), test predictions:
  - Semantic subspaces should have higher SVCCA than random subspaces of same size (already shown in original paper)
  - Categories with more hierarchical structure (like "Calendar" with its year→month→day hierarchy) should show stronger universality than flat categories
  - Correlation patterns within categories should be preserved across models

**Expected Outcomes:**

- Quantification: "Hierarchical structure accounts for ~35% of observed universality, correlation patterns for ~25%, with ~20% from their interaction"
- Structure-aware transfer methods improving matching accuracy by 20-30% over structure-agnostic baselines
- Discovery of which types of semantic structure are most universal (prediction: hierarchies with mutual exclusivity show strongest universality)

---

## 5. The Precision-Recall Trade-off Validates Transformation-Based Similarity

### Implementation Overview

This research direction leverages the precision-recall trade-off observed in SynthSAEBench to understand and optimize transformation-based similarity measures. We implement experiments that show why rotation-invariant measures are necessary and how to optimize them.

### Detailed Implementation Steps

**Task 5.1: Precision-Recall Trade-off Surface Mapping**

Comprehensively map the precision-recall trade-off across all conditions:

- Train SAEs at L0 values: {5, 10, 15, 20, 25, 30, 35, 40, 45, 50} using the L0 controller from SynthSAEBench
- For each L0 and architecture combination, compute:
  - Per-feature precision and recall (treating each SAE feature as a binary classifier for its matched ground-truth feature)
  - Aggregate precision-recall curves
  - F1-score, but also F_β scores for β ∈ {0.5, 2.0} to emphasize precision or recall
- Create a "Pareto frontier" in (precision, recall) space showing the optimal trade-offs achievable by each architecture

**Task 5.2: Basis-Dependent vs Basis-Independent Quality**

Demonstrate that precision-recall trade-offs are basis-dependent while geometric similarity is basis-independent:

- Take two SAEs trained on the same data that achieve different precision-recall trade-offs (e.g., low-L0 SAE with high precision, high-L0 SAE with high recall)
- Apply rotation matrices R to one SAE's features: W_rotated = R·W
- Show that:
  - Precision and recall change dramatically under rotation (basis-dependent)
  - SVCCA and RSA remain constant under rotation (basis-independent)
- This demonstrates why transformation-based measures are necessary for fair comparison

**Task 5.3: Optimal Transformation Learning**

Implement learnable transformations that maximize feature correspondence while respecting geometric structure:

- **Linear Transformation Learning**:
  - Learn T ∈ ℝ^(L×L) that maps features from SAE₁ to SAE₂
  - Objective: maximize MCC(T(W₁), W₂) + λ·SVCCA(T(W₁), W₂)
  - The λ term ensures geometric structure is preserved even while optimizing individual feature matching
- **Non-Linear Transformation Learning**:
  - Use a small neural network T_θ (2-3 layers, ReLU activations)
  - Train on synthetic data where ground truth correspondences are known
  - Test whether non-linear transformations significantly outperform linear ones (hypothesis: only modest improvements for non-linear)
- **Constrained Transformations**:
  - Add constraints preserving specific properties: hierarchy (parent-child relationships), correlation structure, semantic subspace separation
  - Test whether constrained transformations generalize better to out-of-distribution model pairs

**Task 5.4: Transformation Quality Metrics**

Develop comprehensive metrics for evaluating transformation quality:

- **Correspondence Fidelity (CF)**: Using ground truth, measure percentage of correct correspondences after transformation
- **Geometric Preservation (GP)**: Measure how well pairwise distances are preserved: GP = correlation(||w_i - w_j||, ||T(w_i) - T(w_j)||)
- **Semantic Consistency (SC)**: For semantically labeled features, measure whether semantic labels are preserved after transformation
- **Reconstruction Preservation (RP)**: If features are used for steering/intervention, measure whether functionality is preserved: RP = correlation(steering_effect₁, steering_effect₂)
- Create a composite "Transformation Quality Score" (TQS) as weighted combination: TQS = 0.4·CF + 0.3·GP + 0.2·SC + 0.1·RP

**Task 5.5: Trade-off-Aware Matching**

Develop matching algorithms that account for the precision-recall trade-off:

- **Confidence-Weighted Matching**: For each feature pair, compute a confidence score based on activation correlation and precision-recall position
- **Multi-Threshold Matching**: Instead of single-threshold matching, use multiple thresholds and ensemble the results
- **Calibrated Matching**: Use synthetic data to calibrate the relationship between activation correlation and true feature correspondence, then apply calibration to real-world matching

**Task 5.6: L0-Specific Transformation Learning**

Test whether optimal transformations depend on the L0 regime of the SAEs:

- Train SAEs at low L0 (≈15), medium L0 (≈25), high L0 (≈35)
- Learn separate transformations for each L0 regime
- Test cross-regime transfer: Does a transformation learned for low-L0 SAEs work for high-L0 SAEs?
- Hypothesis: "Transformations learned at medium L0 generalize best to other L0 regimes"

**Task 5.7: Real-World Transformation Validation**

Apply transformation learning to real LLM pairs:

- Use Pythia-70m/160m and Gemma-1-2B/2-2B pairs from the universality paper
- Train transformations on layers where universality is known to be strong (e.g., middle layers)
- Validate on held-out layers
- Test whether transformations enable transfer of interpretability artifacts:
  - Extract a steering vector in Model 1
  - Apply learned transformation to predict corresponding steering vector in Model 2
  - Test whether predicted steering vector actually works in Model 2

**Expected Outcomes:**

- Transformation learning improving feature correspondence from baseline 60% to 75-80%
- Demonstration that precision-recall trade-off explains ~40% of why different SAEs learn different features
- Successful transfer of steering vectors across models with >0.7 correlation in steering effects

---

## 6. Dead Latents and Feature Coverage

### Implementation Overview

This research direction addresses the observation that SAEs incompletely cover the feature space, with different SAEs covering different subsets. We implement experiments that characterize feature coverage patterns and develop methods to achieve more complete coverage.

### Detailed Implementation Steps

**Task 6.1: Dead Latent Characterization**

Systematically study dead latent patterns across architectures and conditions:

- Train SAEs across all architectures on SynthSAEBench-16k with multiple seeds
- Track dead latent counts throughout training (checkpoints every 10M tokens)
- For each architecture and condition:
  - **Dead Latent Rate (DLR)**: Percentage of latents that never activate above threshold
  - **Dead Latent Stability (DLS)**: What percentage of latents that are dead at 100M tokens were also dead at 50M tokens? (Measures whether dead latents "die early")
  - **Dead Latent Recovery (DLRec)**: Can we revive dead latents through continued training, auxiliary losses, or re-initialization?
- Analyze which ground-truth features are NOT captured by any SAE features: create a "feature discovery gap" metric

**Task 6.2: Coverage Complementarity Analysis**

Measure the extent to which different SAE architectures cover different feature subsets:

- For each pair of architectures (A, B), compute:
  - **Coverage Overlap (CO)**: Percentage of ground-truth features captured by both A and B
  - **Unique Coverage (UC_A, UC_B)**: Percentage of ground-truth features captured by A but not B (and vice versa)
  - **Union Coverage (UnC)**: Percentage of ground-truth features captured by at least one of A or B
- Hypothesis: "Different architectures have UC ≈ 15-25%, meaning they capture genuinely different features"
- Create visualizations showing which types of features each architecture is best at capturing (e.g., JumpReLU might excel at rare features, Matryoshka at hierarchically organized features)

**Task 6.3: Ensemble Coverage Methods**

Develop methods to combine multiple SAEs for more complete feature coverage:

- **Simple Union Ensemble**: Concatenate features from multiple SAE architectures, remove duplicates based on high correlation (>0.9)
- **Weighted Ensemble**: For each ground-truth feature, identify which SAE architecture captures it best, weight architectures accordingly
- **Complementary Training**: Train SAEs sequentially where each subsequent SAE is encouraged (via auxiliary loss) to capture features missed by previous SAEs
- **Coverage-Aware Architecture Search**: Use synthetic data to identify which architecture combinations achieve highest union coverage with minimal redundancy

**Task 6.4: Active Feature Discovery**

Implement active learning methods that iteratively improve feature coverage:

- Start with a trained SAE (with dead latents)
- Identify dataset examples that are poorly reconstructed (high MSE)
- Analyze these examples to identify which ground-truth features are active but not captured
- Re-initialize dead latents to target these missing features
- Continue training with emphasis on improving reconstruction of poorly-reconstructed examples
- Iterate until coverage plateaus

**Task 6.5: Feature Rarity vs Coverage Analysis**

Test whether feature rarity (firing frequency) predicts coverage failures:

- Divide ground-truth features into deciles by firing frequency
- For each decile, measure what percentage of features are successfully captured (MCC > 0.5) by SAEs
- Hypothesis: "Rare features (bottom 20% by frequency) are captured <50% of the time, while common features (top 20%) are captured >85% of the time"
- Test whether architectural choices differentially affect rare vs. common feature coverage:
  - Prediction: TopK SAEs better at common features due to explicit top-k selection
  - Prediction: L1 SAEs better at rare features due to continuous optimization

**Task 6.6: Coverage Transfer Across Models**

Test whether feature coverage patterns transfer across models:

- Train SAEs on synthetic Model 1, identify which features are consistently missed (across architectures and seeds)
- Train SAEs on synthetic Model 2 (similar but distinct), check if the same types of features are missed
- If patterns transfer, develop "coverage prediction models" that predict which features will be hard to capture based on feature properties (rarity, hierarchy position, correlation with other features)
- Apply predictions to real LLMs: predict which types of features will be poorly covered by SAEs

**Task 6.7: Minimum Coverage Requirements**

Determine what level of feature coverage is sufficient for practical interpretability:

- Create synthetic models where ground truth is known
- Train SAEs achieving various coverage levels (40%, 60%, 80%, 95%)
- Test downstream interpretability tasks with each coverage level:
  - Circuit discovery: Can we identify a ground-truth circuit using SAE features?
  - Steering vector effectiveness: How well do steering vectors work when based on incomplete feature coverage?
  - Causal interventions: What accuracy do we achieve in causal intervention experiments?
- Establish empirically: "X% feature coverage is required for Y% accuracy on task Z"

**Expected Outcomes:**

- Characterization: "Standard training yields 65-75% feature coverage; ensemble methods achieve 85-90% coverage"
- Discovery that dead latents are not random: specific feature types (rare + hierarchically deep + highly correlated) are systematically missed
- Ensemble methods that achieve >85% coverage with only 2-3 architectures (vs. 70% for best single architecture)

---

## 7. Quantitative Validation of Analogous Universality Hypothesis

### Implementation Overview

This research direction provides rigorous quantitative validation of the analogous universality hypothesis using the controlled environment of synthetic benchmarks. We implement comprehensive experiments that measure universality at multiple levels of granularity.

### Detailed Implementation Steps

**Task 7.1: Multi-Level Universality Measurement**

Implement a hierarchy of universality metrics from strongest (identical features) to weakest (unrelated features):

- **Level 1 - Strong Universality (Identical Features)**: Measure percentage of features with MCC > 0.95
- **Level 2 - Feature-Level Analogous Universality**: Percentage with MCC > 0.7 (highly similar but not identical)
- **Level 3 - Geometric Analogous Universality**: Percentage of feature pairs with high SVCCA (>0.6) between their neighborhoods (local geometric structure preserved)
- **Level 4 - Functional Analogous Universality**: Percentage of features with high activation correlation (>0.7) across same dataset
- **Level 5 - Semantic Analogous Universality**: Percentage of features that activate on the same semantic category (even if activation correlation is lower)
- **Level 6 - Weak/No Universality**: Features that fail all above criteria

For each synthetic configuration and real model pair, report the distribution across these levels.

**Task 7.2: Synthetic-to-Real Validity Testing**

Rigorously test whether findings from synthetic data transfer to real LLMs:

- Identify 10 key findings from synthetic experiments (e.g., "middle layers show higher universality than early/late layers")
- For each finding, formulate a testable prediction for real LLMs
- Test predictions on 5 real model pairs: Pythia-70m/160m, Gemma-1-2B/2-2B, Gemma-2-2B/9B, Llama-3/3.1, GPT-2-small/medium
- Compute "prediction accuracy" = percentage of synthetic-derived predictions that hold in real data
- Goal: Achieve >80% prediction accuracy, validating that synthetic benchmarks are realistic

**Task 7.3: Universality Spectrum Experiments**

Map the complete spectrum from no universality to perfect universality:

- Generate synthetic model pairs with controlled universality levels:
  - 0% universality: Completely independent models with no shared features
  - 25% universality: Models share 25% of features, 75% are unique
  - 50% universality: Equal mix of shared and unique features
  - 75% universality: Most features shared, some unique
  - 100% universality: Identical models (perfect universality)
- For each universality level, train SAEs and measure:
  - What SVCCA/RSA scores result from different true universality levels?
  - How does SAE architecture affect the observed universality (e.g., do some architectures "find" more universality)?
- Create calibration curves: Given observed SVCCA=0.6, what is the estimated true universality level (with confidence intervals)?

**Task 7.4: Layer-Wise Universality Patterns**

Validate the layer-wise universality patterns found in the original universality paper:

- Generate synthetic "multi-layer models" where early layers have high superposition, middle layers have moderate superposition, late layers have low superposition but high hierarchy
- Train layer-specific SAEs
- Verify synthetic models reproduce the pattern: "middle layers show highest universality"
- Mechanistically explain why: "Middle layers balance feature distinguishability (enough to avoid superposition confusion) with feature generality (not yet specialized to specific tasks)"

**Task 7.5: Cross-Model-Family Universality**

Test universality across model families (not just within families):

- Generate synthetic models representing different "families":
  - Family A: High superposition, shallow hierarchy
  - Family B: Low superposition, deep hierarchy
  - Family C: Medium superposition, no hierarchy but high correlation
- Train SAEs within and across families
- Measure: Is universality higher within families than across families?
- Prediction: "Within-family universality ≈ 0.7, across-family universality ≈ 0.4"
- Validate on real models: Compare Pythia-to-Pythia vs. Pythia-to-Gemma universality

**Task 7.6: Universality Stability Analysis**

Test whether universality is stable across perturbations:

- Take a model pair with established universality
- Perturb one model: add noise to features, rotate feature basis, change superposition level slightly
- Measure how universality scores degrade with perturbation magnitude
- Establish "universality robustness": How much can models diverge before universality breaks down?
- Application: Predict whether fine-tuned models maintain universality with their base models

**Task 7.7: Quantitative Benchmarking Suite**

Create a comprehensive benchmark suite for universality:

- **Synthetic Universality Benchmark (SUB)**:
  - 20 model pairs covering the full spectrum of universality levels
  - 10 pairs within-family, 10 pairs across-family
  - Ground truth universality levels known
  - Standardized evaluation protocol
- **Real-World Universality Benchmark (RUB)**:
  - 10 real model pairs with established universality patterns from the original paper
  - Standardized SAE training protocols
  - Reference implementations of all universality measures
- Provide leaderboards for:
  - Best SAE architecture for maximizing universality
  - Best transformation learning method
  - Best universality prediction accuracy from synthetic to real

**Expected Outcomes:**

- Quantitative validation: "Analogous universality (levels 2-4) accounts for 60-75% of feature space in typical model pairs"
- Calibrated universality scales allowing statement like: "SVCCA=0.65 corresponds to ~70% true feature overlap"
- Benchmark suite enabling standardized comparison of future universality research

---

## 8. Methodological Alignment

### Implementation Overview

This research direction focuses on unifying and extending the methodologies from both papers into a comprehensive toolkit for studying feature universality. We implement standardized protocols and shared infrastructure.

### Detailed Implementation Steps

**Task 8.1: Unified Feature Matching Pipeline**

Develop a single pipeline that implements all feature matching methods:

- **Module 1: Ground-Truth Matching** (for synthetic data)
  - Input: SAE decoder W, ground-truth features D, correspondence matrix C
  - Output: Ground-truth MCC, recall, precision using known correspondences
- **Module 2: Activation Correlation Matching** (from universality paper)
  - Input: SAE₁ activations A₁, SAE₂ activations A₂ on shared dataset
  - Process: Compute correlation matrix, apply highest-correlation matching or Hungarian algorithm
  - Output: Correlation-based feature pairs, mean correlation score
- **Module 3: Semantic Similarity Matching** (from universality paper's semantic subspace experiments)
  - Input: SAE features with top-k activating tokens, semantic category keywords
  - Process: Match features that share keywords in top-k tokens
  - Output: Semantically matched feature pairs with category labels
- **Module 4: Geometric Matching** (new contribution)
  - Input: SAE₁ decoder W₁, SAE₂ decoder W₂
  - Process: Use SVCCA or learned transformations to find geometric correspondences
  - Output: Geometrically matched feature pairs with alignment quality scores

Make all modules interoperable: output of one module can be input to validation steps in another.

**Task 8.2: Comprehensive Similarity Metrics Library**

Implement all similarity metrics from both papers plus extensions:

- **Reconstruction Metrics**:
  - Explained Variance (R²)
  - Mean Squared Error (MSE)
  - Per-feature reconstruction quality
- **Feature Recovery Metrics** (from SynthSAEBench):
  - Mean Correlation Coefficient (MCC) via Hungarian matching
  - Feature Uniqueness
  - Precision, Recall, F1 per feature and aggregated
- **Representational Similarity Metrics** (from universality paper):
  - SVCCA with configurable dimensionality reduction
  - RSA with configurable inner/outer similarity functions
  - Mean activation correlation
- **Geometric Metrics** (new):
  - Angular similarity distributions
  - Distance preservation scores
  - Neighborhood preservation (k-NN consistency)
- **Semantic Metrics** (from universality paper's semantic experiments):
  - Category coherence scores
  - Semantic subspace SVCCA
  - Cross-category separation scores

Provide implementations in both PyTorch and NumPy for maximum compatibility.

**Task 8.3: Standardized Experimental Protocols**

Create detailed, reproducible protocols for common experimental patterns:

- **Protocol 1: Single-Model SAE Evaluation** (SynthSAEBench style)
  - Training: Specify optimizer, learning rate schedule, batch size, number of tokens
  - Evaluation: Specify metrics, evaluation frequency, dataset splits
  - Provide reference hyperparameters for each SAE architecture
- **Protocol 2: Cross-Model Universality Evaluation** (universality paper style)
  - Feature matching: Specify matching algorithm, threshold values, filtering criteria
  - Similarity measurement: Specify which metrics to use, number of random pairing samples for p-value computation
  - Provide reference configurations for common model pairs
- **Protocol 3: Transformation Learning**
  - Training: Specify architecture, loss function, regularization
  - Validation: Specify held-out test sets, success criteria
- **Protocol 4: Semantic Subspace Analysis**
  - Category definition: Standardized keyword lists
  - Feature assignment: Matching criteria for assigning features to categories
  - Subspace comparison: Statistical testing procedures

**Task 8.4: Shared Infrastructure and Tools**

Build shared infrastructure to enable both research directions:

- **SyntheticModelZoo**: Repository of pre-generated synthetic models with various properties
  - Models covering full parameter space: superposition × hierarchy × correlation
  - Pre-computed ground-truth metadata
  - Versioned and immutable for reproducibility
- **SAE Model Zoo**: Repository of pre-trained SAEs
  - SAEs for all major architectures on standard synthetic models
  - SAEs for common real model pairs (Pythia, Gemma, Llama)
  - Standardized naming and metadata
- **Evaluation Harness**: Unified evaluation framework
  - Load any SAE + any ground-truth or comparison SAE
  - Run all applicable metrics
  - Generate standardized reports and visualizations
- **Experiment Tracking Integration**:
  - Weights & Biases integration for experiment logging
  - Automatic hyperparameter tracking
  - Visualization dashboards for comparing runs

**Task 8.5: Cross-Validation Framework**

Implement methods to validate one paper's findings with the other's methods:

- **Validation 1**: Use SynthSAEBench metrics (MCC, F1) to validate universality paper's claim that high SVCCA indicates good features
  - Train SAEs on synthetic data where ground truth is known
  - Show: "High SVCCA between two SAEs correlates with both SAEs having high MCC to ground truth"
- **Validation 2**: Use universality paper's methods (SVCCA, RSA) to validate SynthSAEBench's finding about MP-SAE overfitting
  - Show: "MP-SAEs have high within-architecture SVCCA (different seeds converge to similar solutions) but lower cross-architecture SVCCA (fundamentally different from other architectures)"
- **Validation 3**: Use both frameworks to validate semantic subspace findings
  - In synthetic data, define ground-truth semantic categories
  - Show: Universality paper's semantic subspace SVCCA scores are higher when SynthSAEBench's HRS (Hierarchy Recovery Score) is also high

**Task 8.6: Reproducibility Infrastructure**

Ensure all research is fully reproducible:

- **Containerization**: Docker containers with all dependencies
- **Configuration Management**: Use Hydra or similar for experiment configuration
- **Data Version Control**: DVC for managing datasets and model checkpoints
- **Continuous Integration**: Automated testing of all pipelines
- **Documentation**: Comprehensive tutorials and API documentation
- **Public Release**: Open-source repository with pre-commit hooks, code formatting, type hints

**Task 8.7: Benchmark Standardization**

Create official benchmark tasks and leaderboards:

- **Task 1: Feature Recovery on SynthSAEBench-16k**
  - Metric: MCC
  - Current best: ~0.75 (Matryoshka SAE)
- **Task 2: Cross-Model Universality on Pythia-70m/160m**
  - Metric: SVCCA at layer pairs
  - Current best: ~0.68 (middle layer pairs)
- **Task 3: Transformation Learning on Synthetic Pairs**
  - Metric: Ground-truth correspondence accuracy after transformation
  - Target: >0.80
- **Task 4: Semantic Subspace Preservation**
  - Metric: Mean category subspace SVCCA
  - Current best: ~0.60

Publish official leaderboards and provide submission infrastructure.

**Expected Outcomes:**

- Unified codebase reducing duplication and enabling direct method comparison
- 10+ standardized protocols enabling reproducible universality research
- Public benchmarks with leaderboards driving community progress
- At least 3 external research groups using the infrastructure within 12 months of release

---

## 9. The Critical Insight: Why Analogous, Not Identical?

### Implementation Overview

This research direction addresses the fundamental theoretical question: Why do SAEs learn analogous rather than identical features? We implement experiments that test mechanistic hypotheses about the origins of feature analogy.

### Detailed Implementation Steps

**Task 9.1: Superposition Decomposition Ambiguity Theory**

Formalize and test the theory that superposition creates fundamental ambiguity in optimal feature decomposition:

- **Theoretical Framework**:
  - Given features in superposition with overlap matrix O where O_ij = |d_i^T d_j| (absolute cosine similarity)
  - Prove: When O is non-identity, there exist multiple factorizations of activation space achieving identical reconstruction error
  - Derive: Number of "equivalent decompositions" as a function of spectral properties of O
- **Empirical Validation**:
  - Generate models with controlled superposition (vary ρₘₘ from 0.05 to 0.30)
  - For each model, train 10 SAE seeds
  - Measure "decomposition diversity" = variance in feature recovery across seeds
  - Test prediction: "Decomposition diversity increases linearly with ρₘₘ"
- **Uniqueness Conditions**:
  - Identify conditions under which decomposition becomes unique
  - Test: "When ρₘₘ < 0.03, different SAE seeds converge to identical features (MCC between seeds > 0.95)"
  - Test: "When ρₘₘ > 0.20, seeds diverge significantly (MCC between seeds < 0.70)"

**Task 9.2: Optimization Landscape Analysis**

Analyze the loss landscape to understand why different training runs find different solutions:

- **Loss Landscape Visualization**:
  - Train two SAE seeds to different solutions (low MCC between them)
  - Interpolate in weight space: W(t) = (1-t)W₁ + t·W₂ for t ∈ [0,1]
  - Plot reconstruction loss, sparsity loss, and ground-truth MCC along this path
  - Hypothesis: "Loss remains low along the entire path (suggesting flat valley connecting multiple good solutions)"
- **Local Minima Characterization**:
  - Use random weight perturbations to estimate local curvature around each solution
  - Measure: Are different solutions in the same basin of attraction or different basins?
  - Test whether adding noise during training increases or decreases decomposition diversity
- **Mode Connectivity**:
  - Apply mode connectivity algorithms to find low-loss paths between solutions
  - If paths exist, this confirms multiple solutions are "equally good" from an optimization perspective

**Task 9.3: Architecture-Specific Biases**

Identify what implicit biases different SAE architectures have that lead them to different solutions:

- **Bias Analysis Framework**:
  - For each architecture, identify its inductive bias (e.g., TopK explicitly biases toward k most important features per sample)
  - Generate synthetic scenarios where different biases would favor different decompositions
  - Example: Create features where half are rare-but-strong and half are common-but-weak. TopK should favor rare-but-strong, while L1 should balance both
- **Systematic Bias Testing**:
  - Create 5 synthetic scenarios favoring different decomposition strategies
  - Train all architectures on each scenario
  - Measure which architectures succeed on which scenarios
  - Build a "bias profile" for each architecture
- **Bias Complementarity**:
  - Test whether architectures with complementary biases can be combined for more complete coverage
  - Example: Combine TopK (good for rare-strong features) + L1 (good for common features) to capture both

**Task 9.4: Information-Theoretic Analysis**

Apply information theory to understand feature analogousness:

- **Mutual Information Framework**:
  - Measure I(SAE_features; ground_truth_features) = mutual information between SAE and ground-truth features
  - Also measure I(SAE₁_features; SAE₂_features) = mutual information between two SAEs trained on same data
  - Hypothesis: "I(SAE₁; ground_truth) ≈ I(SAE₂; ground_truth) but I(SAE₁; SAE₂) < I(SAE₁; ground_truth)" (both SAEs capture same amount of information about ground truth, but encode it differently)
- **Information Bottleneck Perspective**:
  - Frame SAE training as information bottleneck: minimize I(SAE_features; input) while maximizing I(SAE_features; ground_truth_features)
  - Different SAEs may find different trade-offs on this Pareto frontier
  - Measure where different architectures fall on this frontier

**Task 9.5: Causal Mechanisms of Analogy**

Identify causal factors that create analogous vs. identical features:

- **Controlled Intervention Experiments**:
  - Start with a baseline condition yielding analogous features
  - Systematically remove potential causes: eliminate superposition → do features become identical? Eliminate hierarchy → does analogy remain?
  - Build causal graph: superposition → decomposition ambiguity → analogous features
- **Sufficiency and Necessity Tests**:
  - Test sufficiency: "If we have high superposition (ρₘₘ > 0.15), will we always get analogous features?" (Train 20 SAEs, measure if all pairs have MCC < 0.80)
  - Test necessity: "Can we get analogous features without superposition?" (Test on ρₘₘ = 0 models)

**Task 9.6: Analogousness as a Continuum**

Model feature analogousness as a continuous spectrum rather than binary property:

- **Analogousness Score Definition**:
  - Define A(f₁, f₂) = function measuring "degree of analogousness" between features f₁ and f₂
  - Components: activation correlation (functional similarity), geometric alignment (directional similarity), semantic overlap (conceptual similarity)
  - Aggregation: A = 0.4·act_corr + 0.3·cos_sim + 0.3·semantic_overlap
- **Spectrum Analysis**:
  - For all feature pairs from two SAEs, compute analogousness scores
  - Plot distribution of scores: Is it bimodal (analogous vs. non-analogous) or continuous?
  - Hypothesis: "Continuous distribution, suggesting analogousness is a gradual property"
- **Threshold Determination**:
  - Use synthetic data where ground truth is known to calibrate thresholds
  - At what analogousness score do features become "similar enough" for practical purposes?

**Task 9.7: Predictive Models of Analogousness**

Build models that predict the degree of analogousness from model/architecture properties:

- **Feature-Level Predictors**:
  - Input: Feature properties (firing frequency, position in hierarchy, correlation with other features)
  - Output: Expected analogousness to corresponding feature in another model
  - Train on synthetic data, test on real models
- **Architecture-Level Predictors**:
  - Input: SAE architecture types (e.g., "TopK vs. Matryoshka")
  - Output: Expected SVCCA score between the architectures
  - Test: Can we predict which architecture pairs will show highest universality?
- **Model-Level Predictors**:
  - Input: Model properties (size, training data, architecture)
  - Output: Expected cross-model universality (SVCCA, percentage of analogous features)

**Expected Outcomes:**

- Theoretical proof: "Under superposition ρₘₘ > 0.1, optimal decomposition is non-unique"
- Mechanistic understanding: "75% of analogousness stems from superposition ambiguity, 15% from optimization stochasticity, 10% from architectural biases"
- Predictive models achieving R² > 0.65 in predicting analogousness scores
- Clear answer to "Why analogous?": Because superposition creates fundamental ambiguity that different SAEs resolve differently

---

## 10. Synthesis: The Complete Picture

### Implementation Overview

This final research direction integrates all previous findings into a unified theory and comprehensive practical framework. We implement the synthesis phase that combines theoretical understanding with practical tools.

### Detailed Implementation Steps

**Task 10.1: Unified Theory of Feature Universality**

Develop a comprehensive theoretical framework integrating all findings:

- **Theory Document Structure**:
  - **Section 1**: Foundations - Feature representation in neural networks, superposition, dictionary learning
  - **Section 2**: Sources of Non-Uniqueness - Superposition ambiguity, optimization landscape, architectural biases
  - **Section 3**: Spectrum of Universality - Strong (identical) to weak (analogous) to absent (unrelated)
  - **Section 4**: Geometric Preservation - Why rotation-invariant measures capture analogous similarity
  - **Section 5**: Practical Implications - What universality enables for interpretability transfer
- **Mathematical Formalization**:
  - Formalize feature spaces as Riemannian manifolds with induced metric from superposition structure
  - Define universality as geodesic distance between manifolds under optimal diffeomorphism
  - Prove key results about when universality is preserved under transformations
- **Empirical Grounding**:
  - Every theoretical claim must be validated on synthetic data where ground truth is known
  - Provide quantitative bounds: "Under conditions X, universality is at least Y"

**Task 10.2: Practical Universality Toolkit**

Create comprehensive software toolkit implementing all methods:

- **Component 1: Universal Feature Matcher**
  - Input: Two SAE decoders (W₁, W₂)
  - Output: Feature correspondence matrix, confidence scores, analogousness measures
  - Methods: Activation correlation, geometric alignment, semantic matching, learned transformations
  - Selects best method automatically based on data availability and model properties
- **Component 2: Transformation Learning Module**
  - Input: Paired features from two models
  - Output: Learned transformation T and quality metrics
  - Supports: Linear, non-linear, constrained transformations
  - Includes uncertainty quantification
- **Component 3: Transfer Validation Suite**
  - Input: Interpretability artifact from Model 1 (e.g., steering vector), transformation T
  - Output: Transformed artifact for Model 2, predicted effectiveness, validation metrics
  - Tests: Steering effects, circuit functionality, causal interventions
- **Component 4: Universality Predictor**
  - Input: Model properties, SAE architectures, training configuration
  - Output: Predicted universality scores with confidence intervals
  - Trained on extensive synthetic + real-world data

**Task 10.3: Comprehensive Validation Study**

Conduct large-scale validation study integrating all methods:

- **Study Design**:
  - 10 synthetic model pairs covering full parameter space
  - 8 real model pairs: Pythia family, Gemma family, Llama family, GPT-2 family
  - 6 SAE architectures per model
  - All similarity metrics computed
- **Validation Goals**:
  - Test every major claim from the integrated theory
  - Measure consistency between synthetic predictions and real-world outcomes
  - Identify boundary conditions where theory breaks down
- **Multi-Lab Replication**:
  - Coordinate with 3+ external research groups to replicate key findings
  - Ensure methods generalize across research groups and computational environments

**Task 10.4: Application Demonstrations**

Implement concrete applications enabled by universality understanding:

- **Application 1: Zero-Shot Interpretability Transfer**
  - Scenario: Interpret a new LLM by transferring knowledge from an interpreted model
  - Method: Match features using universality toolkit, transfer circuit discoveries
  - Demonstration: Show that 70%+ of circuits transfer successfully
- **Application 2: Cross-Model Steering**
  - Scenario: Develop steering vectors on small model, apply to large model in same family
  - Method: Learn transformations, predict steering vectors for large model
  - Demonstration: Achieve >0.7 correlation in steering effects
- **Application 3: Ensemble Interpretability**
  - Scenario: Combine insights from multiple SAE architectures for more complete understanding
  - Method: Use complementarity analysis to identify which architecture captures which features best
  - Demonstration: Ensemble achieves 85% feature coverage vs. 70% for best single architecture
- **Application 4: Fast Model Comparison**
  - Scenario: Quickly assess similarity between models (for model selection, deployment decisions)
  - Method: Train SAEs, compute universality metrics, interpret similarity scores
  - Demonstration: Accurately predict downstream task transfer with R²>0.6

**Task 10.5: Best Practices Guide**

Develop comprehensive guide for practitioners:

- **Section 1: When to Use Which Method**
  - Decision trees for choosing matching algorithms, similarity metrics, SAE architectures
  - Guidance based on: available compute, model sizes, desired guarantees
- **Section 2: Hyperparameter Recommendations**
  - Validated hyperparameters for different scenarios
  - Sensitivity analysis showing robustness ranges
- **Section 3: Common Pitfalls and Solutions**
  - Failure modes we discovered during research
  - Diagnostic procedures for identifying issues
  - Remediation strategies
- **Section 4: Interpretation Guidelines**
  - How to interpret SVCCA scores, MCC values, analogousness measures
  - What scores mean in practical terms
  - When to trust results, when to be skeptical

**Task 10.6: Educational Materials**

Create materials enabling broader adoption:

- **Tutorial Series**:
  - Tutorial 1: Introduction to Feature Universality (conceptual, no coding)
  - Tutorial 2: Working with Synthetic Benchmarks (hands-on with SynthSAEBench)
  - Tutorial 3: Cross-Model Feature Matching (implementing universality paper methods)
  - Tutorial 4: Transformation Learning (advanced techniques)
  - Tutorial 5: Building Applications (end-to-end application development)
- **Interactive Visualizations**:
  - Web-based demo showing feature matching in action
  - Interactive exploration of synthetic models with controllable parameters
  - Visualization of transformations and their effects
- **Video Content**:
  - Lecture series explaining theoretical foundations
  - Screencasts demonstrating software usage
  - Case studies of successful applications

**Task 10.7: Community Building and Dissemination**

Establish infrastructure for ongoing community engagement:

- **Public Benchmark Challenges**:
  - Annual competition on universality benchmarks
  - Cash prizes for best methods
  - Proceedings publishing winning approaches
- **Reproducibility Repository**:
  - All code, data, trained models openly available
  - Continuous integration ensuring methods remain working
  - Version control with semantic versioning
- **Discussion Forum**:
  - Dedicated space for researchers to discuss universality topics
  - Monthly office hours with research team
  - FAQ and troubleshooting database
- **Academic Dissemination**:
  - Target publications: NeurIPS, ICML, ICLR (main conferences), TMLR (journal)
  - Conference workshops on feature universality
  - Invited talks at major institutions

**Task 10.8: Future Research Directions**

Identify and articulate open problems for the community:

- **Open Problem 1**: Can we achieve >95% feature coverage with better SAE architectures?
- **Open Problem 2**: How does universality extend to multimodal models (vision + language)?
- **Open Problem 3**: Can we characterize universality in RL agents and policy networks?
- **Open Problem 4**: What universal structures exist at the circuit level (beyond features)?
- **Open Problem 5**: How can we use universality for efficient continual learning?

For each problem:

- Clearly define the problem and why it matters
- Provide initial experiments showing feasibility
- Outline potential approaches
- Offer starter code and datasets

**Expected Outcomes:**

- Comprehensive theory paper (target: 40+ pages with appendices)
- Practical toolkit with >1000 GitHub stars within 1 year
- 5+ concrete applications demonstrating real-world value
- Active research community with 20+ groups building on this work
- At least 10 follow-up papers citing this work within 18 months

---

## Conclusion

This integrated research program bridges the gap between controlled synthetic evaluation (SynthSAEBench) and real-world cross-model universality analysis. By implementing the 10 research directions detailed above, we will:

1. **Understand** why SAEs learn analogous rather than identical features
2. **Quantify** the degree of universality across models and conditions
3. **Predict** when and how strongly universality will occur
4. **Exploit** universality for practical interpretability transfer
5. **Advance** the broader field of mechanistic interpretability

The comprehensive implementation plan provided here offers a roadmap for 18-24 months of research that will fundamentally advance our understanding of feature learning in neural networks and enable new capabilities in AI interpretability and safety.

---

## Appendix: Implementation Timeline

**Months 1-4**: Infrastructure (Tasks 1.1-1.3, 2.1, 3.1, 8.1-8.4)  
**Months 5-8**: Core Experiments (Tasks 2.2-2.5, 3.2-3.4, 4.1-4.3, 5.1-5.3)  
**Months 9-12**: Advanced Analysis (Tasks 4.4-4.6, 5.4-5.6, 6.1-6.4, 7.1-7.3)  
**Months 13-16**: Theory & Transfer (Tasks 7.4-7.7, 9.1-9.5, 10.1-10.2)  
**Months 17-20**: Validation & Applications (Tasks 6.5-6.7, 8.5-8.7, 9.6-9.7, 10.3-10.5)  
**Months 21-24**: Synthesis & Dissemination (Tasks 10.6-10.8, paper writing, release)

**Total Estimated Effort**: 4-5 full-time researchers for 24 months

---

**Document End**
