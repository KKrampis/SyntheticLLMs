# Key Papers on Manifold-Aware Sparse Autoencoders

## Research Summary and Literature Review

This document summarizes the key papers discovered through semantic search that are relevant to developing manifold-aware sparse autoencoders based on the SynthSAEBench framework.

---

## 1. Core Papers on Feature Manifolds

### 1.1 Not All Language Model Features Are One-Dimensionally Linear

**Authors:** Joshua Engels, Eric J. Michaud, Isaac Liao, Wes Gurnee, and Max Tegmark
**Publication:** ICLR 2025
**ArXiv:** [https://arxiv.org/abs/2405.14860](https://arxiv.org/abs/2405.14860)
**OpenReview:** [https://openreview.net/forum?id=d63a4AM4hb](https://openreview.net/forum?id=d63a4AM4hb)

**Key Contributions:**

- Provides rigorous definition of **irreducible multi-dimensional features**: features that cannot be decomposed into independent or non-co-occurring lower-dimensional features
- Develops scalable method using sparse autoencoders to automatically find multi-dimensional features
- Discovers **circular features** representing days of the week and months of the year in GPT-2 and Mistral 7B
- Validates computational importance through intervention experiments on Mistral 7B and Llama 3 8B

**Relevance to Our Work:**
This is the foundational paper establishing that manifold features exist in LLMs and are computationally fundamental. Our synthetic benchmark must reproduce these circular structures and test whether SAEs can recover them.

**Key Findings:**

- Circular features form 2D representations: (cos θ, sin θ) for periodic concepts
- These features resist decomposition into 1D components
- Intervention experiments confirm they are the fundamental unit of computation
- Standard SAEs may not optimally represent these geometric structures

---

### 1.2 The Geometry of Concepts: Sparse Autoencoder Feature Structure

### **Authors:** Yuxiao Li, Eric J. Michaud, David D. Baek, Joshua Engels, Xiaoqing Sun, and Max Tegmark **Publication:** Entropy, 27(4), 344 (2025) **ArXiv:** [https://arxiv.org/abs/2410.19750](https://arxiv.org/abs/2410.19750) **MDPI:** [https://www.mdpi.com/1099-4300/27/4/344](https://www.mdpi.com/1099-4300/27/4/344)

**Key Contributions:**

- Analyzes geometric structure of SAE feature dictionaries at three scales:
  1. **Atomic scale:** "Crystal" structures with parallelogram and trapezoid faces
  2. **Intermediate scale:** Spatial modularity resembling functional lobes (math/code features cluster)
  3. **Large scale:** Global organization reflecting semantic relationships
- Demonstrates hierarchical geometric organization in learned features
- Shows that features are not randomly distributed but form structured manifolds

**Relevance to Our Work:**
Provides empirical evidence for geometric structure in SAE features. Our evaluation metrics must capture these multi-scale geometric properties: local crystalline structure, intermediate clustering, and global topology.

**Key Findings:**

- SAE features exhibit non-trivial geometric relationships beyond simple linear directions
- Semantic similarity corresponds to geometric proximity
- Hierarchical organization suggests manifolds participate in hierarchical structures
- Different conceptual domains (math, code, language) form distinct geometric "lobes"

---

### 1.3 Feature Manifold Toy Model

**Authors:** Chris Olah and Josh Batson
**Publication:** Transformer Circuits Thread, May 2023
**URL:** [https://transformer-circuits.pub/2023/may-update/index.html#feature-manifolds](https://transformer-circuits.pub/2023/may-update/index.html#feature-manifolds)

**Key Contributions:**

- Introduces theoretical framework for understanding feature manifolds
- Proposes that related features lie on continuous manifolds where nearby points represent similar concepts
- Contrasts with discrete, independent feature assumption
- Suggests manifolds may be fundamental to neural representation

**Relevance to Our Work:**
Provides theoretical motivation for our manifold-aware synthetic benchmark. The toy model framework can guide our design of synthetic manifolds and help us understand what properties to test.

**Key Concepts:**

- Features form continuous families rather than discrete atoms
- Manifolds allow efficient representation of structured concept spaces
- Standard sparse coding may miss manifold structure
- Need specialized methods to detect and represent manifolds

---

### 1.4 Understanding Sparse Autoencoder Scaling in the Presence of Feature Manifolds

**Authors:** Eric J. Michaud et al.
**Publication:** 2024
**ArXiv:** [https://arxiv.org/abs/2509.02565](https://arxiv.org/abs/2509.02565)
**Goodfire Research:** [https://www.goodfire.ai/research/sae-scaling-with-feature-manifolds](https://www.goodfire.ai/research/sae-scaling-with-feature-manifolds)

**Key Contributions:**

- Adapts capacity-allocation model to understand SAE scaling with manifolds
- Identifies two scaling regimes:
  1. **Pathological regime (β < α):** SAEs learn far fewer features than available latents; continuously "tile" manifolds at expense of discovering rare features
  2. **Benign regime (α < β):** Number of discovered features scales linearly with latents
- Provides theoretical predictions for SAE behavior on manifold data

**Relevance to Our Work:**
**CRITICAL PAPER** for understanding how manifolds affect SAE performance. We must test these scaling predictions empirically on our synthetic benchmark and verify the pathological tiling behavior.

**Key Predictions:**

- SAEs will allocate disproportionately many latents to high-frequency manifolds
- Manifold "tiling" behavior: learning many latents per manifold instead of efficient manifold representation
- Scaling laws differ between manifold and non-manifold features
- Standard SAE architectures may be fundamentally limited for manifold features

---

## 2. Foundational SAE Papers

### 2.1 Towards Monosemanticity: Decomposing Language Models with Dictionary Learning

**Authors:** Trenton Bricken et al. (including Christopher Olah)
**Publication:** Transformer Circuits Thread, 2023
**URL:** [https://transformer-circuits.pub/2023/monosemantic-features](https://transformer-circuits.pub/2023/monosemantic-features)

**Key Contributions:**

- Introduces sparse autoencoder approach for decomposing neural network activations
- Demonstrates that learned features are more monosemantic than raw neurons
- Establishes dictionary learning as interpretability method
- Shows SAEs can find interpretable features in language models

**Relevance to Our Work:**
Foundation for the entire SAE field. Our manifold-aware extensions build on this baseline approach. We must ensure our manifold SAEs maintain the interpretability benefits while improving geometric representation.

---

### 2.2 Scaling and Evaluating Sparse Autoencoders

**Authors:** Leo Gao, Tom Dupré la Tour, et al.
**Publication:** ICLR 2025
**OpenAI Paper:** [https://cdn.openai.com/papers/sparse-autoencoders.pdf](https://cdn.openai.com/papers/sparse-autoencoders.pdf)

**Key Contributions:**

- Comprehensive study of SAE scaling behavior
- Establishes evaluation metrics and benchmarks
- Studies relationship between SAE capacity, sparsity, and reconstruction quality
- Provides practical guidance for training large-scale SAEs

**Relevance to Our Work:**
Provides baseline metrics and scaling laws for standard SAEs. We must compare manifold-aware SAE scaling to these established baselines and show where manifolds create different scaling behavior.

---

### 2.3 Sparse Autoencoders Find Highly Interpretable Features in Language Models

**Authors:** Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, Lee Sharkey
**Publication:** 2023
**ArXiv:** [https://arxiv.org/abs/2309.08600](https://arxiv.org/abs/2309.08600)

**Key Contributions:**

- Demonstrates that SAE features are highly interpretable
- Studies feature activation patterns and semantics
- Validates SAE approach on multiple models
- Analyzes trade-offs between sparsity and reconstruction

**Relevance to Our Work:**
Establishes interpretability as key evaluation criterion. Our manifold-aware SAEs must preserve interpretability while improving geometric representation. We need to test whether manifold features are as interpretable as standard features.

---

## 3. Geometric Representation and Structure

### 3.1 The Linear Representation Hypothesis and the Geometry of Large Language Models

**Authors:** Kiho Park, Yo Joong Choe, Victor Veitch
**Publication:** ICML 2024

**Key Contributions:**

- Formalizes the Linear Representation Hypothesis (LRH)
- Studies geometric properties of concept representations
- Analyzes when linear representations are sufficient
- Provides framework for understanding representation geometry

**Relevance to Our Work:**
Provides theoretical foundation for comparing linear vs. manifold representations. Our synthetic benchmark can instantiate both hypotheses and test which better explains SAE behavior.

---

### 3.2 The Geometry of Categorical and Hierarchical Concepts in Large Language Models

**Authors:** Kiho Park, Yo Joong Choe, Yibo Jiang, Victor Veitch
**Publication:** 2024

**Key Contributions:**

- Studies how categorical and hierarchical concepts are geometrically organized
- Analyzes representation of structured knowledge
- Connects to cognitive science theories of concept representation

**Relevance to Our Work:**
Hierarchical manifolds in our benchmark must reflect these organizational principles. We need to test how SAEs handle the interaction between hierarchy and manifold structure.

---

### 3.3 Calendar Feature Geometry in GPT-2 Layer 8 Residual Stream SAEs

**Authors:** Patrick Leask, Bart Bussmann, Neel Nanda
**Publication:** LessWrong, August 2024
**URL:** [https://www.lesswrong.com/posts/WsPyunwpXYCM2iN6t](https://www.lesswrong.com/posts/WsPyunwpXYCM2iN6t)

**Key Contributions:**

- Empirical discovery of geometric structure in calendar features
- Shows days and months form 2D geometric patterns
- Provides visualization techniques for feature geometry
- Validates existence of structured manifolds in real LLMs

**Relevance to Our Work:**
Direct empirical validation that calendar manifolds exist in real models. Our synthetic benchmark should reproduce this specific phenomenon and test whether SAEs can recover it. This provides a concrete validation target.

---

## 4. SAE Architecture Variants

### 4.1 Improving Dictionary Learning with Gated Sparse Autoencoders

**Authors:** Senthooran Rajamanoharan et al.
**Publication:** 2024
**ArXiv:** [https://arxiv.org/abs/2404.16014](https://arxiv.org/abs/2404.16014)

**Key Contributions:**

- Introduces gated SAE architecture separating feature selection from magnitude
- Improves reconstruction quality and feature interpretability
- Shows benefits of architectural modifications

**Relevance to Our Work:**
Gating mechanism could be adapted for manifold SAEs: gate could select which manifold is active, then magnitude determines position on manifold.

---

### 4.2 Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders

**Authors:** Senthooran Rajamanoharan et al.
**Publication:** 2024
**ArXiv:** [https://arxiv.org/abs/2407.14435](https://arxiv.org/abs/2407.14435)

**Key Contributions:**

- Introduces JumpReLU activation for better reconstruction
- Reduces reconstruction-sparsity trade-off
- Demonstrates architecture improvements matter

**Relevance to Our Work:**
Could be combined with manifold-aware architectures to improve reconstruction while maintaining manifold structure.

---

## 5. Evaluation and Analysis Methods

 Representational Similarity Analysis

**Authors:** Nikolaus Kriegeskorte, Marieke Mur, Peter Bandettini
**Publication:** Frontiers in Systems Neuroscience, 2008
**DOI:** [10.3389/neuro.06.004.2008](https://www.frontiersin.org/articles/10.3389/neuro.06.004.2008/full)

**Key Contributions:**

- Framework for comparing neural representations across systems
- Distance-based similarity metrics
- Applicable to both biological and artificial systems

**Relevance to Our Work:**
RSA methods can be adapted to compare manifold structures between ground truth and learned SAE representations. Provides principled framework for manifold alignment metrics.

---

### 5.2 Similarity of Neural Network Representations Revisited

**Authors:** Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton
**Publication:** 2019
**ArXiv:** [https://arxiv.org/abs/1905.00414](https://arxiv.org/abs/1905.00414)

**Key Contributions:**

- Comprehensive comparison of representation similarity metrics
- Studies CKA, CCA, and other methods
- Provides guidance on metric selection

**Relevance to Our Work:**
These metrics can be adapted for comparing manifold representations. We need manifold-aware versions of CKA/CCA to evaluate geometric structure preservation.

---

## 6. Feature Phenomena in SAEs

### 6.1 A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders

**Authors:** David Chanin et al.
**Publication:** 2024
**ArXiv:** [https://arxiv.org/abs/2409.14507](https://arxiv.org/abs/2409.14507)

**Key Contributions:**

- Identifies **feature splitting**: one ground-truth feature learned by multiple SAE latents
- Identifies **feature absorption**: multiple ground-truth features collapsed to one SAE latent
- Studies frequency and causes of these phenomena

**Relevance to Our Work:**
**Critical for predictions:** We predict manifold features will exhibit more splitting (SAE tiles manifold) and less absorption (manifold resists collapse). Must measure these phenomena on our synthetic benchmark.

---

### 6.2 Decomposing the Dark Matter of Sparse Autoencoders

**Authors:** Joshua Engels, Logan Riggs, Max Tegmark
**Publication:** 2024
**ArXiv:** [https://arxiv.org/abs/2410.14670](https://arxiv.org/abs/2410.14670)

**Key Contributions:**

- Studies "dark matter": activation variance not explained by learned features
- Analyzes what SAEs miss
- Provides insights into SAE limitations

**Relevance to Our Work:**
Manifolds may be part of the "dark matter" that standard SAEs fail to capture. Our manifold-aware SAEs should reduce dark matter by explicitly modeling geometric structure.

---

## 7. Neuroscience Connections

### 7.1 Bipartite Invariance in Mouse Primary Visual Cortex

**Authors:** Zhiwei Ding et al.
**Publication:** bioRxiv, 2023
**DOI:** [10.1101/2023.03.15.532836](https://www.biorxiv.org/content/10.1101/2023.03.15.532836v1)

**Key Contributions:**

- Studies manifold representations in biological neural systems
- Shows visual cortex represents features on low-dimensional manifolds
- Demonstrates manifolds are fundamental to biological computation

**Relevance to Our Work:**
Provides biological precedent for manifold representations. If biological systems use manifolds, artificial systems likely do too. Our work bridges neuroscience and AI interpretability.

---

### 7.2 Alignment of Brain Embeddings and Artificial Contextual Embeddings

**Authors:** Ariel Goldstein et al.
**Publication:** Nature Communications, 15(1):2768, 2024
**DOI:** [10.1038/s41467-024-46631-y](10.1038/s41467-024-46631-y)

**Key Contributions:**

- Shows brain and LLM representations share common geometric patterns
- Demonstrates alignment between biological and artificial embeddings
- Suggests universal geometric principles

**Relevance to Our Work:**
If brain and LLM representations share geometry, manifold structures may be universal. Our synthetic benchmark can test whether SAEs capture these universal geometric principles.

---

## 8. Theoretical Foundations

### 8.1 Toy Models of Superposition

**Authors:** Nelson Elhage et al.
**Publication:** Transformer Circuits Thread, 2022
**URL:** [https://transformer-circuits.pub/2022/toy_model/index.html](https://transformer-circuits.pub/2022/toy_model/index.html)

**Key Contributions:**

- Theoretical framework for understanding superposition
- Toy models demonstrating when and why superposition occurs
- Analysis of feature interference and recovery

**Relevance to Our Work:**
Superposition interacts with manifolds (Michaud et al.). We must extend toy model framework to include manifold features and test predictions about manifold superposition.

---

### 8.2 Sparse Coding with an Overcomplete Basis Set: A Strategy Employed by V1?

**Authors:** Bruno A. Olshausen and David J. Field
**Publication:** Vision Research, 37(23), 1997
**DOI:** [10.1016/S0042-6989(97)00169-7](https://www.sciencedirect.com/science/article/pii/S0042698997001697)

**Key Contributions:**

- Original sparse coding framework for neural representation
- Demonstrates biological plausibility of sparse representations
- Establishes connection to visual cortex

**Relevance to Our Work:**
Historical foundation for sparse autoencoders. Our manifold extensions maintain this biological inspiration while adding geometric structure.

---

## 9. Practical Implications

### 9.1 Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignment

**Authors:** Harrish Thasarathan et al.
**Publication:** 2025
**ArXiv:** [https://arxiv.org/abs/2502.03714](https://arxiv.org/abs/2502.03714)

**Key Contributions:**

- Studies universality of SAE features across models
- Analyzes concept alignment between different architectures
- Tests transferability of learned representations

**Relevance to Our Work:**
If manifolds are fundamental, they should be universal across models. Our validation experiments should test whether manifold structures transfer across different synthetic models and to real LLMs.

---

#### 9.2 Steering Language Models with Activation Engineering

**Authors:** Alexander Matt Turner et al.
**Publication:** 2024
**ArXiv:** [https://arxiv.org/abs/2308.10248](https://arxiv.org/abs/2308.10248)

**Key Contributions:**

- Methods for steering LLM behavior through activation manipulation
- Demonstrates practical applications of interpretability
- Shows importance of understanding representation geometry

**Relevance to Our Work:**
Understanding manifolds enables better steering: can move along geodesics on manifolds for smoother, more natural interventions. Manifold-aware steering could be more effective than linear steering.

---

## 10. Research Gaps and Our Contributions

### Identified Gaps:

1. **No synthetic benchmark for manifold features** → We create ManifoldSynthSAEBench
2. **No manifold-specific evaluation metrics** → We develop geodesic preservation, topology preservation, curvature accuracy
3. **No manifold-aware SAE architectures** → We propose GL-SAE, MP-SAE, HM-SAE, ALM-SAE
4. **No systematic hypothesis testing** → We generate models for LRH, manifold hypothesis, etc.
5. **No scaling laws for manifolds** → We test Michaud et al.'s predictions empirically
6. **No transfer learning from synthetic to real** → We test manifold SAE transfer to GPT-2

### Our Novel Contributions:

1. **Extension of SynthSAEBench** with circular, spherical, toroidal manifolds
2. **Manifold-aware evaluation suite** with 6 new metrics
3. **Four novel SAE architectures** explicitly modeling manifolds
4. **Representation hypothesis testing framework** with testable predictions
5. **Validation against real LLM phenomena** (calendar features, circular structure)
6. **Practical implementation plan** with 7 phases over 20 weeks

---

## 11. Integration with Existing Work

### How Our Work Connects:

**Engels et al. (2025)** discovers circular features → We create synthetic circular features with known ground truth → We test if SAEs can recover them

**Li et al. (2025)** finds geometric structure → We generate synthetic geometric structure → We measure if SAEs preserve it

**Michaud et al. (2024)** predicts pathological scaling → We test predictions empirically on controlled data → We validate or refute theory

**Olah & Batson (2023)** proposes manifold toy model → We implement full-scale realistic benchmark → We bridge theory to practice

**SynthSAEBench** provides LRH baseline → We add manifold features → We compare LRH vs. manifold hypotheses

### Synthesis:

Our work **synthesizes** theoretical insights (Olah, Michaud), empirical discoveries (Engels, Li, Leask), and rigorous benchmarking (SynthSAEBench, Gao) into a **unified framework** for studying manifold-aware interpretability.

---

## 12. Next Steps

### Immediate Actions:

1. ✅ Create research plan document
2. ✅ Organize key papers in Manifold-SAEs folder
3. ✅ Summarize literature and identify gaps
4. ⏳ Begin Phase 1 implementation: Manifold generation code
5. ⏳ Implement circular and spherical manifolds
6. ⏳ Generate first dataset with 10 manifolds

### Long-term Roadmap:

- **Months 1-2:** Implement data generation and evaluation metrics
- **Months 3-4:** Develop manifold-aware SAE architectures
- **Months 5-6:** Run hypothesis testing experiments
- **Months 7-8:** Validate against real LLMs
- **Months 9-10:** Write paper and release code/data

---

## Conclusion

This literature review identifies **feature manifolds** as a critical but understudied aspect of neural representation. Recent empirical work (Engels, Li, Leask) provides strong evidence that manifolds exist in real LLMs and are computationally fundamental. Theoretical work (Olah, Michaud) predicts that standard SAEs will struggle with manifolds due to pathological scaling and tiling behaviors.

**Our contribution:** A systematic, rigorous approach to understanding manifold-aware interpretability through controlled synthetic experiments. By extending SynthSAEBench, we can test competing hypotheses, develop better architectures, and establish new evaluation standards.

**Impact:** This work has the potential to reshape how the field thinks about feature learning, representation geometry, and the fundamental units of neural computation. If manifolds are indeed fundamental, then manifold-aware interpretability tools are not optional—they are essential for understanding how language models truly work.

---

## Papers Included in Folder

### PDFs Available:

1. `18572_SynthSAEBench_Evaluating.pdf` - Original SynthSAEBench paper
2. `Sparse_Autoencoders_Universal_Feature_Spaces.pdf` - Universal SAE features
3. `SAE_Research_Analysis.pdf` - General SAE research overview

### Papers to Download:

1. Engels et al. (2025) - Not All Language Model Features Are Linear
2. Li et al. (2025) - Geometry of Concepts
3. Michaud et al. (2024) - SAE Scaling with Manifolds
4. Gao et al. (2025) - Scaling and Evaluating SAEs
5. Bricken et al. (2023) - Towards Monosemanticity
6. Cunningham et al. (2023) - SAEs Find Interpretable Features
7. Rajamanoharan et al. (2024a) - Gated SAEs
8. Rajamanoharan et al. (2024b) - JumpReLU SAEs
9. Park et al. (2024) - Linear Representation Hypothesis
10. Chanin et al. (2024) - Feature Splitting and Absorption

---

**Last Updated:** February 11, 2026
**Status:** Literature review complete, ready for implementation Phase 1
