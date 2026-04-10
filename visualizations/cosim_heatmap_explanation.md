**Figure.** Pairwise cosine similarity heatmap for SAE features organized into MITRE ATLAS taxonomy trees. Rows and columns correspond to the same set of features, ordered breadth-first within each tree (root, then level-1 children, then level-2 leaves). Colored rectangles outline level-1 subtree blocks. The pronounced columnar banding within each block reflects the shared parent-vector component inherited by sibling features under the same tactic node.

---

## Columnar Symmetry in the Cosine Similarity Heatmap

The heatmap displays pairwise cosine similarities between SAE feature vectors organized according to MITRE ATLAS tactic hierarchies. Features within the same subtree share a common parent direction because each child feature is constructed as a linear combination of its parent vector and an orthogonal component: *d*_child = α·*d*_parent + β·*d*_⊥. As a result, any two siblings *i* and *j* under the same parent have cosine similarity with an arbitrary third feature *x* that is approximately equal — cos(*d*_i, *x*) ≈ cos(*d*_j, *x*) — since both are dominated by the same α·*d*_parent term. This makes entire columns within a subtree block appear nearly uniform in color, producing the characteristic vertical stripes visible inside the colored bounding boxes.

The contrast between blocks of different colors illustrates the hierarchical structure at a coarser level. Features from different tactics have lower mutual similarities because their respective parent vectors are mutually orthogonal by construction, attenuating the shared-component effect across subtree boundaries. The columnar symmetry thus serves as a geometric fingerprint of the construction rule: the more features share ancestry, the more their similarity profiles with the rest of the matrix align, collapsing distinct rows into visually indistinguishable columns.

---

## How the Figure Was Generated

The heatmaps are produced by `tutorials/semantic_dictionary_tutorial.ipynb`. What follows describes each stage of that notebook.

### Taxonomy and Semantic Dictionary

The starting point is `feature_hierarchies/mitre_atlas_adversarial_ml.json`, a hand-authored semantic dictionary encoding the MITRE ATLAS adversarial machine learning threat taxonomy as a forest of concept trees. The file contains 13 root nodes — one per ATLAS tactic, including *Adversarial Evasion*, *Data Poisoning*, *Model Extraction and Stealing*, and *Prompt Injection and LLM Exploitation* — each heading a subtree of depth 2. Internal nodes represent tactic families (e.g. *Physical Domain Evasion*, *Digital Domain Evasion*) and leaves represent individual techniques (e.g. *Adversarial Patch*, *Black-Box Evasion*, *Stop Sign Attack*). Across all 13 trees the taxonomy defines 148 hierarchical feature slots, with individual tree sizes ranging from 10 to 13 nodes.

Each node in the JSON carries a `label`, an `alpha` (α) value specifying the desired cosine similarity between the node's feature vector and its parent's, a `beta` (β) value satisfying α² + β² = 1, a `mutually_exclusive_children` flag indicating whether sibling features are treated as alternatives by the firing sampler, and a `children` array that recurses into the same schema. Root nodes are assigned α = 0 and β = 1, meaning they have no inherited direction and are initialised as free unit vectors.

### Synthetic Model Construction

A `SyntheticModel` is instantiated with 512 features embedded in a 128-dimensional hidden space, giving a 4× superposition ratio. Because the number of features exceeds the ambient dimensionality, the model cannot represent all features in an orthogonal basis and must instead place them in superposition — an arrangement that mirrors the situation hypothesised for features in large language models. The model is seeded to be reproducible and configured with a `HierarchyConfig` pointing at the JSON taxonomy, which activates the semantic geometry initialiser.

The initialiser traverses the JSON forest in depth-first order. For each parent–child edge it constructs the child's feature vector according to the rule *d*_child = α·*d*_parent + β·*d*_⊥, where *d*_⊥ is a unit vector in the subspace orthogonal to *d*_parent computed by Gram–Schmidt orthogonalisation. This guarantees cos(*d*_child, *d*_parent) = α exactly, so the α values in the JSON are not targets to be learned but rather geometric invariants enforced at initialisation time. The remaining 364 features not covered by the taxonomy are assigned mutually orthogonalised random unit vectors, filling the residual capacity of the hidden space as uniformly as the dimensionality allows.

### Geometric Verification

Before any training, the notebook verifies the constructed geometry numerically by iterating over all 135 parent–child pairs in the hierarchy and computing the cosine similarity between each pair of normalised feature vectors. The maximum absolute deviation from the corresponding α value is below 2×10⁻⁷ across all pairs — a level consistent with single-precision floating-point rounding — confirming that the Gram–Schmidt procedure realises the intended similarities to essentially exact precision. This verification is not merely illustrative: it establishes a ground truth against which the SAE's learned representations can later be compared, and it guards against numerical drift that could otherwise invalidate the theoretical guarantees of the construction.

### Feature Firing Distribution

The model generates activations by sampling a sparse binary mask over the 512 features for each token in a batch, then computing the hidden state as the sum of the activated feature vectors scaled by sampled magnitudes. Firing probabilities are drawn from a Zipfian distribution with exponent 0.5, clipped to the interval [10⁻³, 0.3], so that a small number of features fire frequently while the majority fire rarely. This power-law structure reflects empirical observations about feature frequency in large language models and introduces a realistic diversity of sample counts across features.

The hierarchy additionally enforces a conditional firing rule: a child feature may only be active on a given token if its parent is also active. This constraint ensures that the semantic relationships encoded in the JSON are reflected not only in the geometry of the feature vectors but also in their co-occurrence statistics. The notebook verifies this constraint exhaustively over 10,000 sampled batches, finding zero violations. Under these conditions the average number of active features per token (L0) is approximately 2.7, reflecting the combined effect of the low base firing rates and the hierarchical gating that suppresses children whenever their parent is inactive.

### Cosine Similarity Heatmaps

For each of four randomly selected tactic trees the heatmap is constructed as follows. First, all nodes in the tree are enumerated in breadth-first order — root, then level-1 children in left-to-right order, then level-2 leaves grouped under their respective level-1 parents. This ordering is consequential: it places nodes that share a level-1 ancestor in contiguous index ranges, which is precisely what makes the block structure visible. The model's feature dictionary matrix *W* of shape (512, 128) is then L2-normalised row-wise to produce unit vectors, and the cosine similarity matrix for the selected nodes is computed as the inner product of the corresponding submatrix with its own transpose, yielding a symmetric *n*×*n* matrix.

The entries of this matrix have a predictable structure that follows directly from the construction rule. Diagonal entries are identically 1. Parent–child entries equal the child's α value by the geometric invariant established above. Sibling entries — between two leaves that share a level-1 parent — are elevated above zero because both vectors contain the term α·*d*_parent; the expected similarity between two such siblings with mixing coefficients (α₁, β₁) and (α₂, β₂) is approximately α₁α₂, modulated by the degree to which their respective orthogonal components happen to align. Cross-subtree entries — between features from different level-1 blocks — are close to zero because the corresponding parent vectors are mutually orthogonal, so the shared-direction term that elevates within-block similarities is absent across block boundaries.

Finally, for each level-1 child of the root, the set of all descendant feature indices is recovered via `get_all_feature_indices()` and mapped back to the BFS ordering, yielding a contiguous index range. A colored rectangle is drawn over the corresponding diagonal block to make the grouping explicit, with distinct colors assigned to each level-1 subtree in sequence.

### SAE Training and Recovery

The main experiment in the notebook trains a BatchTopK Sparse Autoencoder on hidden activations sampled from the synthetic model. The encoder projects the 128-dimensional hidden state into a 512-dimensional latent space, retains the top-*k* = 15 activations per token, and the decoder reconstructs the hidden state from the sparse latent code. With 10 million training tokens, batch size 1024, and learning rate 3×10⁻⁴, training converges in roughly two minutes on CPU.

The choice of *k* = 15 against a true L0 of 2.7 is deliberately generous: the SAE is permitted to use substantially more active latents per token than the data actually contains, which biases it toward high recall at the expense of precision. After training, recovery is evaluated by matching each SAE latent to its closest ground-truth feature via cosine similarity in activation space, then computing standard binary classification metrics over the resulting assignment. The results are summarised below.

| Metric | Value |
|:---|---:|
| Explained variance (R²) | 0.971 |
| Matthews Correlation Coefficient | 0.732 |
| Uniqueness | 0.773 |
| F1 score | 0.475 |
| Precision | 0.372 |
| Recall | 0.783 |
| SAE L0 | 15.0 |
| True L0 | 2.7 |
| Dead latents | 72 / 512 |
| Shrinkage | 0.985 |

The high explained variance (0.97) confirms that the SAE reconstructs the hidden state faithfully in terms of mean squared error. The recall of 0.78 indicates that most ground-truth features have a corresponding SAE latent that activates when the feature is present. The precision of 0.37, however, reflects a substantial rate of spurious activations — latents firing in the absence of any corresponding ground-truth feature — which is the expected consequence of the 5.6× mismatch between the SAE's permitted L0 and the true signal sparsity. Of the 512 SAE latents, 72 remain permanently inactive throughout training, likely corresponding to features so rare under the Zipfian schedule that the SAE never encounters enough positive examples to learn them. The shrinkage near 1.0 confirms that the BatchTopK constraint prevents the magnitude collapse that afflicts L1-penalised autoencoders.

The connection to the heatmaps is direct. The block-diagonal similarity structure visible in each panel represents exactly the geometric correlations the SAE must disentangle. When a leaf feature shares α = 0.65 with its level-1 parent, their hidden representations differ by a rotation of only arccos(0.65) ≈ 49°, making it genuinely difficult for the encoder to resolve whether the parent, the child, or both were active on a given token. This ambiguity is the proximate cause of the precision shortfall, and it grows with α: trees whose level-1 groups have large mixing coefficients will show more tightly clustered blocks in the heatmap and correspondingly lower precision in the SAE recovery metrics.
