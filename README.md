# Geometric Feature Invariance in SAEs: A Framework for Transferable Mechanistic Interpretability and Scalable AI Safety

## Abstract

This research will integrate controlled synthetic model features evaluation with cross-model geometric feature invariance analysis, towards developing a principled framework for transferable interpretability across model families and variants. Recent work has showed that Sparse Autoencoders (SAEs) trained on different LLMs learn geometrically similar feature spaces (invariance and analogous feature universality) while exhibiting trade-offs against synthetic ground-truth features, meaning no current SAE architecture can perfectly recover the ground-truth features.

Our work will address fundamental challenges in mechanistic interpretability by establishing principled methods for leveraging geometric feature universality. Efficient feature transfer is critical for AI safety because it enables:

1. Rapid safety evaluation of new models without restarting interpretability analysis from scratch
2. Early detection of hazardous capabilities by comparing feature spaces to known dangerous configurations
3. Reliable monitoring across deployment contexts by tracking feature drift
4. Scalable oversight of large model families where per-model analysis becomes infeasible

During the project implementation we will characterize geometric patterns that remain stable across models, and geometric transformation methods that reliably map feature correspondences, validated against synthetic ground-truth. Furthermore, we will demonstrate that safety-relevant interventions transfer within the same family of models, or fine-tuned variants of a model. This research will accelerate mechanistic interpretability, and enable efficient safety analysis as AI systems grow in capability and complexity.

---

## Theory of Change

| | |
|---|---|
| **Activities** | Develop synthetic benchmarks to test SAE transfer and the presence of invariant feature structures |
| **Outputs** | Protocols that enable reliable cross-model interpretability transfer based on invariant AI safety-related features and circuits |
| **Outcomes** | Enable interpretations and safety interventions developed on one model to reliably transfer to other models in the same family |
| **Impact** | AI safety becomes scalable without requiring complete re-interpretation and SAE generation for each new model, fine-tuned variant, or model family member |
| **Key Assumption** | Geometric feature similarity is both necessary and sufficient for interpretability transfer |
