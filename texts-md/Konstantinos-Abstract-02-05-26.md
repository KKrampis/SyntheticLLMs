# Research Proposal Abstract

**Title: "From Analogous Features to Transferable Interpretability: A Synthetic-Validated Framework for Universal SAE Feature Geometry"**

This research program integrates controlled synthetic evaluation with cross-model universality analysis to develop a principled framework for 
transferable interpretability in large language models. Recent work shows that sparse autoencoders (SAEs) trained on different LLMs learn 
geometrically similar feature spaces (analogous feature universality) and exhibit systematic architectural trade-offs against ground-truth features (SynthSAEBench), yet no existing work addresses the critical question: can we design SAE training procedures that maximize feature transferability across models?

We propose to develop multi-model synthetic benchmarks with known feature correspondences, enabling rigorous evaluation of transformation methods that map feature spaces between models. By systematically varying superposition levels, hierarchical structures, and correlation patterns, we will identify which geometric aspects are preserved across models and which SAE architectures best capture transferable properties. These insights will inform transfer-optimized SAE training protocols, validated through improved cross-model transfer of steering vectors and circuit interpretations on real LLM pairs.

This work addresses fundamental challenges in mechanistic interpretability and AI safety by establishing principled methods for leveraging geometric feature universality. Efficient feature transfer is critical for AI safety because it enables: (1) rapid safety evaluation of new models without restarting interpretability analysis from scratch, (2) early detection of hazardous capabilities by comparing feature spaces to known dangerous configurations, (3) reliable monitoring across deployment contexts by tracking feature drift, and (4) scalable oversight of large model families where per-model analysis becomes infeasible.

Expected outcomes include: (1) clear characterization of geometric patterns that remain stable across models, (2) transformation methods that reliably map feature correspondences validated against synthetic ground-truth, (3) improved SAE training guidelines prioritizing cross-model compatibility, and (4) demonstrations that safety-relevant interventions transfer from smaller to larger models in the same family. This research will accelerate mechanistic interpretability, enable efficient safety analysis of model families, and provide scalable methods for monitoring AI systems as they grow in capability and complexity.



### Theory of Change

Activities: We will develop multi-model synthetic benchmarks where multiple synthetic LLMs with known but different feature decompositions generate training data, enabling controlled testing of whether SAEs trained on one synthetic model can transfer to another through the geometric transformations identified by universality via invariant transformations.

Outputs: This produces validated methods for predicting which SAE architectural configurations will produce maximally transferable features, formalized as synthetic-to-real transfer protocols that map from controlled benchmark results to real LLM deployment.

Outcomes: These protocols enable safety-relevant interpretations and interventions developed on one model to reliably transfer to other models in the same family, validated through synthetic benchmarks that directly test the hypothesis that feature space similarity enables cross-model transfer of interpretability techniques.

Impact: AI safety analysis becomes scalable without requiring complete re-interpretation for each new model, fine-tuned variant, or model family member, accelerating our ability to identify and mitigate risks across the rapidly expanding landscape of deployed language models.

Key Assumption: Geometric feature similarity, as measured through invariant transformations, is both necessary and sufficient for interpretability transfer between models, such that controlling for geometric properties in SAE training reliably produces transferable safety-relevant features.
