# mitre_atlas_adversarial_ml

**Source:** MITRE ATLAS (Adversarial Threat Landscape for Artificial-Intelligence Systems)

MITRE ATLAS Adversarial ML Taxonomy — semantic dictionary for SAELens synthetic model hierarchy. Each tree encodes a top-level ATLAS tactic or attack family. Alpha values reflect semantic closeness to the parent concept; siblings use distinct alpha values so their feature vectors do not collapse. Beta = sqrt(1 - alpha^2) preserves unit norm. Root nodes use alpha=0.0, beta=1.0.

> Nodes show `α` (semantic similarity to parent). ⊕ = mutually exclusive children.

## Adversarial Evasion

```mermaid
graph TD
    n0["Adversarial Evasion\nα=0.00 ⊕"]
    n1["Physical Domain Evasion\nα=0.65"]
    n0 --> n1
    n2["Adversarial Patch\nα=0.65"]
    n1 --> n2
    n3["Physical World Perturbation\nα=0.55"]
    n1 --> n3
    n4["Stop Sign Attack\nα=0.50"]
    n1 --> n4
    n5["Digital Domain Evasion\nα=0.55"]
    n0 --> n5
    n6["White-Box Evasion\nα=0.65"]
    n5 --> n6
    n7["Black-Box Evasion\nα=0.55"]
    n5 --> n7
    n8["Transfer-Based Evasion\nα=0.50"]
    n5 --> n8
    n9["Semantic Evasion\nα=0.45"]
    n0 --> n9
    n10["Paraphrase Attack\nα=0.65"]
    n9 --> n10
    n11["Synonym Substitution\nα=0.55"]
    n9 --> n11
```

## Adversarial Perturbation Crafting

```mermaid
graph TD
    n0["Adversarial Perturbation Crafting\nα=0.00 ⊕"]
    n1["Lp-Norm Bounded Perturbation\nα=0.65"]
    n0 --> n1
    n2["L-infinity FGSM Attack\nα=0.65"]
    n1 --> n2
    n3["PGD Attack\nα=0.60"]
    n1 --> n3
    n4["L2 Carlini-Wagner Attack\nα=0.55"]
    n1 --> n4
    n5["DeepFool Attack\nα=0.50"]
    n1 --> n5
    n6["Score-Based Black-Box Attack\nα=0.55"]
    n0 --> n6
    n7["NES Gradient Estimation\nα=0.65"]
    n6 --> n7
    n8["ZOO Attack\nα=0.55"]
    n6 --> n8
    n9["Decision-Based Attack\nα=0.45"]
    n0 --> n9
    n10["Boundary Attack\nα=0.65"]
    n9 --> n10
    n11["HopSkipJump Attack\nα=0.55"]
    n9 --> n11
```

## Data Poisoning

```mermaid
graph TD
    n0["Data Poisoning\nα=0.00 ⊕"]
    n1["Training Data Manipulation\nα=0.65"]
    n0 --> n1
    n2["Label Flipping\nα=0.65"]
    n1 --> n2
    n3["Feature Poisoning\nα=0.55"]
    n1 --> n3
    n4["Availability Attack\nα=0.50"]
    n1 --> n4
    n5["Targeted Integrity Poisoning\nα=0.55"]
    n0 --> n5
    n6["Gradient-Aligned Poisoning\nα=0.65"]
    n5 --> n6
    n7["Witches Brew Attack\nα=0.55"]
    n5 --> n7
    n8["Federated Poisoning\nα=0.45"]
    n0 --> n8
    n9["Byzantine Gradient Attack\nα=0.65"]
    n8 --> n9
    n10["Model Replacement Attack\nα=0.55"]
    n8 --> n10
    n11["Free-Rider Attack\nα=0.45"]
    n8 --> n11
```

## Backdoor and Trojan Attacks

```mermaid
graph TD
    n0["Backdoor and Trojan Attacks\nα=0.00 ⊕"]
    n1["Static Trigger Backdoor\nα=0.65"]
    n0 --> n1
    n2["Visible Patch Trigger\nα=0.65"]
    n1 --> n2
    n3["Invisible Steganographic Trigger\nα=0.55"]
    n1 --> n3
    n4["Frequency Domain Trigger\nα=0.50"]
    n1 --> n4
    n5["Dynamic Trigger Backdoor\nα=0.55"]
    n0 --> n5
    n6["Input-Conditioned Trigger\nα=0.65"]
    n5 --> n6
    n7["Sample-Specific Trigger\nα=0.55"]
    n5 --> n7
    n8["Instruction Backdoor\nα=0.45"]
    n0 --> n8
    n9["Sleeper Agent Backdoor\nα=0.65"]
    n8 --> n9
    n10["Task-Specific Trojan\nα=0.55"]
    n8 --> n10
```

## Model Extraction and Stealing

```mermaid
graph TD
    n0["Model Extraction and Stealing\nα=0.00 ⊕"]
    n1["Query-Based Model Extraction\nα=0.65"]
    n0 --> n1
    n2["Functionally Equivalent Extraction\nα=0.65"]
    n1 --> n2
    n3["Task-Specific Model Stealing\nα=0.55"]
    n1 --> n3
    n4["Knowledge Distillation Attack\nα=0.50"]
    n1 --> n4
    n5["Architecture Inference\nα=0.55"]
    n0 --> n5
    n6["Side-Channel Timing Inference\nα=0.65"]
    n5 --> n6
    n7["Hyperparameter Extraction\nα=0.55"]
    n5 --> n7
    n8["Intellectual Property Theft\nα=0.45"]
    n0 --> n8
    n9["Watermark Evasion\nα=0.65"]
    n8 --> n9
    n10["API Scraping for Training\nα=0.50"]
    n8 --> n10
```

## Membership Inference

```mermaid
graph TD
    n0["Membership Inference\nα=0.00 ⊕"]
    n1["Black-Box Membership Inference\nα=0.65"]
    n0 --> n1
    n2["Confidence-Score Attack\nα=0.65"]
    n1 --> n2
    n3["Loss-Threshold Attack\nα=0.55"]
    n1 --> n3
    n4["Shadow Model Inference\nα=0.50"]
    n1 --> n4
    n5["White-Box Membership Inference\nα=0.55"]
    n0 --> n5
    n6["Gradient Norm Attack\nα=0.65"]
    n5 --> n6
    n7["Intermediate Representation Analysis\nα=0.50"]
    n5 --> n7
    n8["Attribute Inference\nα=0.45"]
    n0 --> n8
    n9["Demographic Attribute Leakage\nα=0.65"]
    n8 --> n9
    n10["Sensitive Feature Reconstruction\nα=0.50"]
    n8 --> n10
```

## Model Inversion and Data Reconstruction

```mermaid
graph TD
    n0["Model Inversion and Data Reconstruction\nα=0.00 ⊕"]
    n1["Output-Based Inversion\nα=0.65"]
    n0 --> n1
    n2["Confidence Vector Inversion\nα=0.65"]
    n1 --> n2
    n3["Generative Model Inversion\nα=0.55"]
    n1 --> n3
    n4["Gradient-Based Inversion\nα=0.55"]
    n0 --> n4
    n5["Training Data Reconstruction\nα=0.65"]
    n4 --> n5
    n6["Feature Attribution Inversion\nα=0.55"]
    n4 --> n6
    n7["Gradient Inversion in Federated Learning\nα=0.50"]
    n4 --> n7
    n8["LLM Training Data Extraction\nα=0.45"]
    n0 --> n8
    n9["Memorization Extraction\nα=0.65"]
    n8 --> n9
    n10["Verbatim Training Data Elicitation\nα=0.55"]
    n8 --> n10
```

## Prompt Injection and LLM Exploitation

```mermaid
graph TD
    n0["Prompt Injection and LLM Exploitation\nα=0.00 ⊕"]
    n1["Direct Prompt Injection\nα=0.65"]
    n0 --> n1
    n2["Instruction Override\nα=0.65"]
    n1 --> n2
    n3["Role-Playing Jailbreak\nα=0.55"]
    n1 --> n3
    n4["System Prompt Extraction\nα=0.50"]
    n1 --> n4
    n5["Indirect Prompt Injection\nα=0.55"]
    n0 --> n5
    n6["Web Content Hijacking\nα=0.65"]
    n5 --> n6
    n7["RAG Context Poisoning\nα=0.55"]
    n5 --> n7
    n8["Tool Output Injection\nα=0.50"]
    n5 --> n8
    n9["LLM Agent Manipulation\nα=0.45"]
    n0 --> n9
    n10["Agentic Task Hijacking\nα=0.65"]
    n9 --> n10
    n11["Privilege Escalation via LLM\nα=0.55"]
    n9 --> n11
    n12["Multi-Agent Prompt Relay Attack\nα=0.45"]
    n9 --> n12
```

## ML Reconnaissance

```mermaid
graph TD
    n0["ML Reconnaissance\nα=0.00 ⊕"]
    n1["ML API Probing\nα=0.65"]
    n0 --> n1
    n2["Input Space Exploration\nα=0.65"]
    n1 --> n2
    n3["Output Distribution Profiling\nα=0.55"]
    n1 --> n3
    n4["Confidence Score Harvesting\nα=0.50"]
    n1 --> n4
    n5["Architecture Fingerprinting\nα=0.55"]
    n0 --> n5
    n6["Timing-Based Fingerprinting\nα=0.65"]
    n5 --> n6
    n7["Layer Count Estimation\nα=0.55"]
    n5 --> n7
    n8["Dataset Reconnaissance\nα=0.45"]
    n0 --> n8
    n9["Training Distribution Inference\nα=0.65"]
    n8 --> n9
    n10["Data Source Identification\nα=0.55"]
    n8 --> n10
```

## ML Supply Chain Attack

```mermaid
graph TD
    n0["ML Supply Chain Attack\nα=0.00 ⊕"]
    n1["Dataset Compromise\nα=0.65"]
    n0 --> n1
    n2["Web Scraping Poisoning\nα=0.65"]
    n1 --> n2
    n3["Benchmark Contamination\nα=0.55"]
    n1 --> n3
    n4["Crowdsourcing Manipulation\nα=0.50"]
    n1 --> n4
    n5["Pretrained Model Compromise\nα=0.55"]
    n0 --> n5
    n6["Malicious Pretrained Weights\nα=0.65"]
    n5 --> n6
    n7["Hugging Face Model Hub Poisoning\nα=0.55"]
    n5 --> n7
    n8["ML Framework and Dependency Attack\nα=0.45"]
    n0 --> n8
    n9["Malicious Python Package Injection\nα=0.65"]
    n8 --> n9
    n10["ML Pipeline Code Injection\nα=0.55"]
    n8 --> n10
    n11["GPU Driver Exploitation\nα=0.40"]
    n8 --> n11
```

## Adversarial Robustness Degradation

```mermaid
graph TD
    n0["Adversarial Robustness Degradation\nα=0.00 ⊕"]
    n1["Defense Bypass\nα=0.65"]
    n0 --> n1
    n2["Adversarial Training Bypass\nα=0.65"]
    n1 --> n2
    n3["Certified Defense Evasion\nα=0.55"]
    n1 --> n3
    n4["Preprocessing Defense Bypass\nα=0.50"]
    n1 --> n4
    n5["Obfuscation-Based Evasion\nα=0.55"]
    n0 --> n5
    n6["Input Transformation Obfuscation\nα=0.65"]
    n5 --> n6
    n7["Gradient Masking Exploitation\nα=0.55"]
    n5 --> n7
    n8["Adaptive Attack Strategy\nα=0.45"]
    n0 --> n8
    n9["Defense-Aware Perturbation\nα=0.65"]
    n8 --> n9
    n10["Ensemble Attack for Robustness\nα=0.55"]
    n8 --> n10
```

## ML Model Access and Initial Compromise

```mermaid
graph TD
    n0["ML Model Access and Initial Compromise\nα=0.00 ⊕"]
    n1["Direct Model Access\nα=0.65"]
    n0 --> n1
    n2["Public API Exploitation\nα=0.65"]
    n1 --> n2
    n3["Stolen Credentials for ML Platform\nα=0.55"]
    n1 --> n3
    n4["Physical Access to Model Files\nα=0.40"]
    n1 --> n4
    n5["Indirect Model Access via Surrogate\nα=0.55"]
    n0 --> n5
    n6["Shadow Model Training\nα=0.65"]
    n5 --> n6
    n7["Transfer Learning Abuse\nα=0.50"]
    n5 --> n7
    n8["ML Infrastructure Exploitation\nα=0.45"]
    n0 --> n8
    n9["MLOps Pipeline Hijacking\nα=0.65"]
    n8 --> n9
    n10["Cloud GPU Instance Compromise\nα=0.50"]
    n8 --> n10
```

## ML Exfiltration

```mermaid
graph TD
    n0["ML Exfiltration\nα=0.00 ⊕"]
    n1["Model Weight Exfiltration\nα=0.65"]
    n0 --> n1
    n2["Direct Checkpoint Download\nα=0.65"]
    n1 --> n2
    n3["Covert Exfiltration via Side Channel\nα=0.50"]
    n1 --> n3
    n4["Training Data Exfiltration\nα=0.55"]
    n0 --> n4
    n5["Data Pipeline Interception\nα=0.65"]
    n4 --> n5
    n6["Membership Inference-Driven Exfiltration\nα=0.50"]
    n4 --> n6
    n7["Model Behavior Exfiltration\nα=0.45"]
    n0 --> n7
    n8["Systematic Output Logging\nα=0.65"]
    n7 --> n8
    n9["Capability Probing for Theft\nα=0.55"]
    n7 --> n9
```
