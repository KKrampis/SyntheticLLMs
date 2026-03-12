# Geometric Feature Invariance in SAEs: A Framework for Transferable Mechanistic Interpretability and Scalable AI Safety

**Authors:** Konstantinos Krampis  
**Affiliation:** [Institution]  
**Date:** March 3, 2026

**Targeted output:** NeurIPS, deadline April 06 - May 7th 2027.

## Abstract

This research will integrate controlled synthetic model features evaluation with cross-model geometric feature invariance analysis, towards developing a principled framework for transferable interpretability across model families and variants. Recent work has showed that Sparse Autoencoders (SAEs) trained on different LLMs learn geometrically similar feature spaces (invariance and analogous feature universality) while exhibiting trade-offs against synthetic ground-truth features, meaning no current SAE architecture can perfectly recover the ground-truth features. Our work will address fundamental challenges in mechanistic interpretability by establishing principled methods for leveraging geometric feature universality. Efficient feature transfer is critical for AI safety because it enables rapid safety evaluation of new models without restarting interpretability analysis from scratch, early detection of hazardous capabilities by comparing feature spaces to known dangerous configurations, reliable monitoring across deployment contexts by tracking feature drift, and scalable oversight of large model families where per-model analysis becomes infeasible. During the project implementation we will characterize geometric patterns that remain stable across models, and geometric transformation methods that reliably map feature correspondences, validated against synthetic ground-truth. Furthermore, we will demonstrate that safety-relevant interventions transfer within the same family of models, or fine-tuned variants of a model. This research will accelerate mechanistic interpretability, and enable efficient safety analysis as AI systems grow in capability and complexity.

## Theory of Change

Our approach involves developing synthetic benchmarks to test SAE transfer and the presence of invariant feature structures, which will produce protocols that enable reliable cross-model interpretability transfer based on invariant AI safety-related features and circuits. This will enable interpretations and safety interventions developed on one model to reliably transfer to other models in the same family, ultimately making AI safety scalable without requiring complete re-interpretation and SAE generation for each new model, fine-tuned variant, or model family member. The key assumption underlying this work is that geometric feature similarity is both necessary and sufficient for interpretability transfer.

## 1. Related Work

Sparse autoencoders have emerged as a powerful tool for decomposing neural network activations into interpretable features, yet measuring the quality of these features remains an active area of research. Recent work has converged on a multi-dimensional evaluation framework that addresses different aspects of SAE quality: concept detection, feature separability, reconstruction fidelity, and human interpretability [1]. The development of comprehensive evaluation suites has revealed that gains on traditional proxy metrics, such as reconstruction loss at a given sparsity level, do not reliably translate to better practical performance, highlighting the need for more sophisticated assessment methods.

Sparse probing methodology has proven particularly valuable for testing whether individual SAE features capture known concepts [2]. This technique selects the top-k SAE features most correlated with binary classification tasks and trains logistic regression probes on only those features. When applied to tasks spanning language identification, profession classification, and sentiment analysis across 35 binary classification tasks, sparse probing reveals whether SAEs successfully learn dedicated, interpretable representations of specific concepts. However, recent comprehensive testing across 113 datasets has shown that SAE-based sparse probes do not consistently outperform simple baselines like logistic regression on raw activations for supervised classification tasks [3].

Concept disentanglement represents another crucial dimension of SAE evaluation, measuring whether these models properly separate independent features into non-overlapping representations [4]. Originally developed using language models trained on chess and Othello game transcripts where ground-truth features are formally specifiable, this approach has been extended to general language models through metrics such as RAVEL, spurious correlation removal, and targeted probe perturbation. These evaluations test whether intervening on specific latents can change one attribute without altering others, and whether concept-specific latent sets remain non-overlapping. Notably, Matryoshka SAEs exhibit positive scaling on disentanglement metrics as dictionary size grows, while most other architectures show degraded performance due to feature splitting.

Automated interpretability techniques have revolutionized the scalability of SAE feature evaluation by using large language models to generate and assess natural language explanations of feature behavior [5]. This two-stage pipeline involves an explainer model producing concise descriptions based on activating examples, followed by a scorer model predicting feature activation patterns based on these explanations. The accuracy of these predictions serves as a measure of explanation quality and feature interpretability. While this approach has dramatically reduced evaluation costs and enabled analysis of millions of features, it often struggles to differentiate between SAE architectures, suggesting it captures necessary but not sufficient aspects of SAE quality.

The integration of these evaluation methods has revealed that SAE quality is irreducibly multi-dimensional, with no single metric capable of identifying optimal architectures [1]. This finding has shifted the field away from optimizing purely for reconstruction loss toward embracing nuanced, multi-metric evaluation paradigms. The challenge of transferable interpretability across model families adds another layer of complexity, as geometric feature similarity patterns must be characterized and validated against synthetic ground-truth to enable reliable cross-model feature correspondence and safety-relevant intervention transfer.

## 2. Methodology

### 2.1 Synthetic Data Generation Framework

Our methodology builds upon the SynthSAEBench framework [6], which generates large-scale synthetic activation data with realistic neural network feature characteristics. The synthetic data generation process begins by defining N = 16,384 ground-truth features in a D = 768-dimensional activation space. Each feature i is represented by a unit direction vector $\mathbf{d}_i \in \mathbb{R}^{768}$, created through random sampling from a standard normal distribution followed by L2 normalization: $\mathbf{d}_i = \mathbf{g}_i / \|\mathbf{g}_i\|_2$ where $\mathbf{g}_i \sim \mathcal{N}(0, I_{768})$. To reduce spurious correlations, these direction vectors undergo an orthogonalization process that minimizes $L_{ortho} = \sum_{i \neq j} (\mathbf{d}_i^T \mathbf{d}_j)^2 + \lambda \sum_i (\|\mathbf{d}_i\|_2 - 1)^2$, pushing vectors toward orthogonality while maintaining unit length.

The framework incorporates three critical phenomena observed in real neural networks: superposition, hierarchy, and correlation. Superposition arises naturally when the number of features (N = 16,384) exceeds the activation dimensionality (D = 768), forcing features to share representational space. This overlap is quantified using Mean Max Cosine Similarity: $\rho_{mm} = \frac{1}{N} \sum_{i=1}^N \max_{j \neq i} |\mathbf{d}_i^T \mathbf{d}_j|$, where $\rho_{mm} \approx 0.15$ indicates moderate superposition in the standard benchmark. Hierarchical structure is implemented as a forest of 128 trees with branching factor 4 and maximum depth 3, encompassing 10,884 features with parent-child dependencies enforced through coefficient constraints: $c_{child} \leftarrow c_{child} \cdot \mathbf{1}[c_{parent} > 0]$.

### 2.2 Feature Dictionary Construction and Correlation Structure

The synthetic activation generation follows the linear superposition model where each sample activation is constructed as $\mathbf{a} = \sum_{i=1}^N c_i \mathbf{d}_i$, with coefficients $c_i$ sampled from hierarchically-constrained distributions. Each feature i receives a firing probability $p_i$ drawn from a Zipfian distribution to model realistic feature frequency patterns, with firing magnitudes $\mu_i$ linearly interpolated from 27.0 to 18.0 across the frequency spectrum. Standard deviations $\sigma_i$ follow a folded normal distribution to capture realistic activation variance patterns observed in language models.

To model realistic feature co-occurrence patterns while maintaining computational tractability, the framework employs a low-rank factorization of the correlation matrix $\Sigma = \mathbf{F}\mathbf{F}^T + \text{diag}(\boldsymbol{\delta})$. This reduces memory requirements from 268 million entries to 1.6 million entries—a 163× reduction compared to storing the full correlation matrix. The correlation scale parameter controls the strength of feature dependencies, with typical values around 0.075 producing realistic but manageable correlation patterns.

Superposition emerges naturally from the dimensional constraints, as packing N = 16,384 features into D = 768 dimensions necessitates overlap. The degree of feature interference scales approximately as $O(1/\sqrt{D})$ with increasing dimensionality, while growing with the number of packed features. For example, increasing from 4,096 to 16,384 features raises $\rho_{mm}$ from ~0.12 to ~0.18 without orthogonalization, demonstrating how representational crowding intensifies with feature density. This superposition creates fundamental ambiguity in activation interpretation: a given activation vector $\mathbf{a} = [1.9, 0.436]$ could arise from multiple coefficient combinations when features overlap significantly, such as $c_1=1, c_2=1$ versus $c_1=2, c_2=0$ for features with directions $\mathbf{d}_1 = [1, 0]$ and $\mathbf{d}_2 = [0.9, 0.436]$.

### 2.3 Hierarchical Constraint Enforcement and Tree Structure

The hierarchical structure implements realistic concept taxonomies where child features can only activate when their parent features are active, mimicking how abstract concepts enable more specific subcategories. The enforcement algorithm processes constraints level-by-level from roots to leaves: for each node with parent index $parent_{idx}$ and child index $child_{idx}$, if $c_{parent_{idx}} = 0$ then $c_{child_{idx}} = 0$. Additionally, mutual exclusion among siblings ensures that only one child within a sibling group can fire simultaneously, creating competitive dynamics that reflect real-world concept relationships.

This hierarchical distribution creates a realistic mix of 10,884 features subject to conceptual constraints alongside 5,500 independent features. The tree structure follows a balanced branching pattern: 128 root features at depth 0, 512 features at depth 1, 2,048 at depth 2, and 8,192 leaf features at depth 3. This distribution models how natural language concepts organize from broad categories (like "animal") through intermediate levels ("mammal", "dog") to specific instantiations ("golden retriever"), providing a controlled testbed for evaluating whether sparse autoencoders can recover and respect compositional semantic structure embedded in neural representations.

## 3. Experimental Setup

### 3.1 Addressing Semantic Structure Limitations in Representation Space

The fundamental challenge in evaluating sparse autoencoders on semantic structures lies in the gap between statistical and semantic hierarchies. Current synthetic benchmarks, while powerful for controlled experimentation, implement purely statistical dependencies that lack semantic meaning. The core limitation is that semantics requires compositional structure in the representation itself, not merely correlated firing probabilities. In real language models, semantic relationships like "deceptive reasoning" and "goal misrepresentation" exhibit meaningful connections because the activation patterns for child concepts literally contain components of parent concepts, creating genuine compositional structure where child representations are built from parent representations plus additional information.

In contrast, existing synthetic frameworks treat hierarchical features as independent random directions with statistical dependency rules. When a child feature fires, it adds an arbitrary vector to the activation that bears no compositional relationship to its parent's contribution. This disconnect means that while we can test whether SAEs handle hierarchical statistical dependencies, we cannot evaluate whether they recover semantically meaningful compositional structures. The semantic labels assigned to features ("Deceptive Reasoning," "Goal Misrepresentation") serve only human interpretation—the synthetic model possesses no semantic understanding of these concepts.

### 3.2 Semantic Hierarchy Requirements and Practical Implementation Approaches

Implementing genuinely semantic hierarchies in synthetic evaluation frameworks would require fundamental architectural changes beyond current capabilities. Semantic basis vectors with interpretable meaning would replace random orthogonal directions, defining fundamental semantic concepts like intentionality, truthfulness, goal-oriented behavior, and theory of mind as basis vectors, then constructing complex features as meaningful combinations of these primitives.

Given the fundamental paradox that creating semantically meaningful synthetic features requires knowing what semantic structure looks like in representation space—precisely what we aim to investigate—we implement three practical approaches. The weak semantic structure approach creates statistical signatures that correlate with potential semantic patterns, while the hybrid approach leverages real model guidance by collecting LLM activations on curated datasets with known semantic properties. The explicit limitation acknowledgment approach recognizes that synthetic benchmarks test statistical decomposition capabilities rather than semantic understanding, focusing on measurable properties: handling hierarchical statistical dependencies, decomposing correlated versus independent features, and scaling behavior under various statistical constraints.

### 3.3 Compositional Feature Directions with Hierarchical Constraints

We introduce partial semantic structure by modifying how feature direction vectors are constructed within hierarchical relationships, making child feature directions compositionally dependent on their parent directions rather than having them be independent random vectors. When constructing a hierarchical feature tree, we define each child feature direction as $\mathbf{d}_{child} = \alpha \cdot \mathbf{d}_{parent} + \beta \cdot \mathbf{d}_{\perp}$, where $\mathbf{d}_{parent}$ is the parent's direction vector, $\mathbf{d}_{\perp}$ is a component orthogonal to the parent representing the specialization of the child concept, and $\alpha, \beta$ are mixing coefficients that control how much of the parent's representation is inherited.

To obtain the orthogonal component $\mathbf{d}_{\perp}$, we employ Gram-Schmidt orthogonalization starting with a random vector $\mathbf{v}$  and subtracting its projection onto the parent direction: $\mathbf{d}_{\perp} = \mathbf{v} - (\mathbf{v} \cdot \mathbf{d}_{parent}) \mathbf{d}_{parent}$. The coefficient$ $  $\beta$ controls how much orthogonal specialization enters the final child direction—a large $\beta$ relative to $\alpha$ means the child direction tilts strongly toward its unique subspace, while a small $\beta$ means the child remains closely aligned with the parent direction. This creates genuine compositional structure where activations containing child features literally contain components of parent features in their vector representations.

The theoretical foundation rests on semantic relatedness manifesting as geometric structure in representation space. When we set $\alpha > 0$, we create non-zero cosine similarity between parent and child feature directions: $\cos(\theta) = \mathbf{d}_{child}^T \mathbf{d}_{parent} = \alpha$ after normalization. This creates testable predictions: SAEs that successfully decompose these features should discover latents where decoder directions for child features have high cosine similarity with decoder directions for parent features, and interventions that ablate parent features should impair reconstruction of child features more severely than unrelated features.

### 3.4 LLM-Generated Misalignment Hierarchies and Geometric Implementation

Our experimental pipeline leverages large language models to generate interpretable concept hierarchies with explicit semantic similarity values, then translates these into geometric constraints within the synthetic framework. In the first step, we prompt an LLM to produce a tree of misalignment-related concepts with "Deceptive Reasoning" as a root, children like "Goal Misrepresentation" and "Information Withholding," and grandchildren such as "Reward Hacking" and "Sycophantic Agreement." Critically, we ask the LLM to assign an $\alpha$ value to each parent-child edge encoding its judgment of semantic similarity—"Reward Hacking" might receive $\alpha = 0.4$ from "Goal Misrepresentation" since it represents a specific instantiation, while "Goal Misrepresentation" might receive $\alpha = 0.7$ from "Deceptive Reasoning" as a direct sub-case.

In the second step, we translate the tree into feature directions by starting at the root with a random unit vector $\mathbf{d}_{root}$ and recursively computing child directions using the compositional formula. The mixing coefficient $\beta$ is derived from $\alpha$ to achieve the specified cosine similarity after normalization. Grandchildren inherit geometric structure transitively from both parents and grandparents—"Reward Hacking" contains directional overlap with both "Goal Misrepresentation" and "Deceptive Reasoning," capturing the correct semantic property of compositional concept inheritance.

The resulting feature directions integrate directly into SynthSAEBench's hierarchy mechanism, where the parent firing-probability constraint $c_{child} \leftarrow c_{child} \cdot \mathbf{1}[c_{parent} > 0]$ enforces statistical dependencies while the geometric construction ensures that hidden activations for child concept samples literally contain components pointing toward parent concept directions. This enables precise diagnostic evaluation: we can test whether SAEs learn latents whose decoder directions maintain high cosine similarity with ground-truth concept directions, whether ablating latents aligned with "Deceptive Reasoning" impairs reconstruction of "Reward Hacking" more than unrelated features, and whether SAEs correctly split hierarchies or inappropriately absorb child concepts into parent latents.

## 4. Results

[Results generation plan]

 **Week 1:** Mar. 4 – Mar. 8, set up SynthSAEBench and start running sample data.

 **Week 2:** Mar. 9 – Mar. 15, domain knowledge hierachy set, encoding into hierarchy following theoretical framework described in earlier sections, initial SAE runs.

 **Week 3:** Mar. 16 – Mar. 22, a set of parametrized runs, output colletion and visualization. 

 **Week 4:** Mar. 23 – Mar. 27, results section writing and discussion, overall polishing of manuscript.

## 5. Discussion

[Discussion section to be developed]

## 6. Conclusion

### 6.1 Implications for AI Safety Research

The limitations of statistical versus semantic hierarchies carry significant implications for AI safety applications of sparse autoencoder research. Current synthetic evaluation frameworks enable testing of necessary but insufficient conditions for safety-relevant interpretability. We can reliably assess whether SAEs can detect rare but statistically structured patterns, which matters if scheming behaviors exhibit characteristic correlation signatures. The frameworks also allow evaluation of scaling behavior when dangerous features are sparse, robustness of detection mechanisms to feature correlations and hierarchical dependencies, and whether transfer learning approaches work effectively for statistical signatures of concerning behaviors.

However, these capabilities come with critical limitations for safety applications. We cannot test whether SAEs actually understand what concepts like "deception" mean compositionally, whether ablating supposed "deception features" prevents deceptive reasoning in functionally meaningful ways, whether the hierarchical relationships we construct in synthetic data match real semantic hierarchies in deployed models, or whether synthetic representations of concerning behaviors bear any resemblance to how actual models represent such capabilities.

For practical AI safety research, this analysis suggests using synthetic benchmarks like SynthSAEBench to test necessary conditions while recognizing their insufficiency. If SAEs fail on statistical hierarchies, they will certainly fail on real semantic hierarchies. However, success on statistical hierarchies does not guarantee success on semantic ones, requiring additional validation on real models with genuine semantic content. The synthetic approach provides a controlled environment for understanding SAE scaling properties, architectural trade-offs, and fundamental limitations, but cannot substitute for evaluation on semantically grounded data where safety-relevant behaviors emerge naturally from model training rather than statistical construction.

This framework suggests a two-stage validation approach: first, demonstrate SAE capabilities on increasingly complex synthetic statistical structures to establish baseline competence, then validate these capabilities on real model activations with known safety-relevant semantic content. Only architectures that succeed at both stages should be considered reliable for safety-critical interpretability applications, ensuring that statistical decomposition capabilities translate to meaningful understanding of dangerous model behaviors.

## Acknowledgments

[Acknowledgments section to be developed]

## References

[1] Karvonen, A., Loughran, T., & Kenton, Z. (2025). SAEBench: A comprehensive evaluation suite for sparse autoencoders. *arXiv preprint arXiv:2503.09532*.

[2] Gurnee, W., Nanda, N., Pauly, M., Harvey, K., Troitskii, D., & Bertsimas, D. (2023). Finding neurons in a haystack: Case studies with sparse probing. *arXiv preprint arXiv:2305.01610*.

[3] Kantamneni, S., Wattenberg, M., & Firat, O. (2025). Evaluating sparse autoencoder representations with supervised classification tasks. *arXiv preprint arXiv:2502.16681*.

[4] Karvonen, A., Foote, B., Bricken, T., & Brown, T. (2024). SAEs learn sparse representations of sparse representations. *arXiv preprint arXiv:2408.00113*.

[5] Paulo, F., Friedman, D., & EleutherAI. (2025). Delphi: Towards machine psychology through conditional language modeling. *arXiv preprint arXiv:2410.13928*.

[6] Chanin, D., et al. (2026). SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data. *arXiv preprint arXiv:2602.14687*.

## Appendices

### Appendix A: Additional Experimental Details

[Additional details to be developed]

### Appendix B: Supplementary Results

[Supplementary results to be developed]