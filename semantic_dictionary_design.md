# Semantic Dictionary Design: Hierarchical Feature Geometry

This document describes the changes needed to implement a custom semantic dictionary
matching the hierarchy described in https://kkrampis.github.io/SyntheticLLMs/.

---

## Overview

The paper defines a **hierarchical forest** of 128 trees (branching factor 4, max depth 3,
~10,880 hierarchical features out of 16,384 total) where parent-child relationships are
encoded geometrically:

```
d_child = α · d_parent + β · d_⊥
```

- `α` (semantic similarity coefficient): cosine similarity between child and parent directions
- `β = sqrt(1 - α²)`: orthogonal mixing weight
- `d_⊥`: orthogonal component unique to the child (via Gram-Schmidt)
- After normalization: `cos(θ_child_parent) = α`

Children only activate when their parent is active (`c_child ← c_child · 1[c_parent > 0]`)
and siblings are mutually exclusive. This is already partially supported by the codebase —
the gaps are in the **geometric initialization** of feature vectors and **per-edge α/β values**.

The full semantic structure — labels, α, β, and tree topology — is loaded from a **JSON file**
at runtime, making the taxonomy user-configurable without code changes.

---

## How the JSON Populates the Dictionary

The `"trees"` array in the JSON defines however many hierarchical trees the user wants —
there is no requirement to fill all 128 slots. `generate_hierarchy()` reads the JSON,
assigns feature indices sequentially to all nodes across all trees, and stops. The remaining
`num_features − N_hierarchical` features are **free/non-hierarchical**: they exist in the
`FeatureDictionary` and `ActivationGenerator` but appear in no `HierarchyNode`. They fire
independently (no parent-child gating) and their vectors are orthogonalized against each
other using `orthogonalize_embeddings()`. The JSON does not need to enumerate these slots.

For example, with `num_features=16384` and a JSON file containing 50 trees of depth 3,
branching factor 4 (50 × 85 = 4,250 hierarchical nodes):

```
Features 0–4,249    → hierarchical, geometric α/β initialization
Features 4,250–16,383 → free, orthogonalized initialization
```

A complete 4-ary tree of depth 3 has 1 + 4 + 16 + 64 = **85 nodes per tree**
(geometric sum: (4⁴ − 1)/(4 − 1) = 85). 128 such trees cover **10,880 hierarchical
features**, leaving 5,504 free features for a 16,384-feature model.

`load_semantic_dictionary()` counts nodes in the JSON and raises a `ValueError` if the
total exceeds `num_features`.

---

## JSON Schema for the Semantic Dictionary

The JSON file is the single source of truth for the hierarchy. Each tree is a nested object.
Example with two trees (`semantic_dictionary.json`):

```json
{
  "trees": [
    {
      "label": "Deceptive Reasoning",
      "alpha": 0.0,
      "beta": 1.0,
      "mutually_exclusive_children": true,
      "children": [
        {
          "label": "Goal Misrepresentation",
          "alpha": 0.7,
          "beta": 0.714,
          "mutually_exclusive_children": false,
          "children": [
            { "label": "Framing Manipulation", "alpha": 0.6, "beta": 0.8,   "children": [] },
            { "label": "Selective Omission",   "alpha": 0.5, "beta": 0.866, "children": [] }
          ]
        },
        {
          "label": "Reward Hacking",
          "alpha": 0.4,
          "beta": 0.917,
          "mutually_exclusive_children": false,
          "children": [
            { "label": "Proxy Gaming",               "alpha": 0.6, "beta": 0.8,   "children": [] },
            { "label": "Specification Exploitation", "alpha": 0.5, "beta": 0.866, "children": [] }
          ]
        }
      ]
    },
    {
      "label": "Sycophancy",
      "alpha": 0.0,
      "beta": 1.0,
      "mutually_exclusive_children": true,
      "children": [
        { "label": "Approval Seeking",  "alpha": 0.65, "beta": 0.76,  "children": [] },
        { "label": "Opinion Mirroring", "alpha": 0.55, "beta": 0.835, "children": [] }
      ]
    }
  ]
}
```

Fields per node:

| Field | Type | Description |
|-------|------|-------------|
| `label` | string | Human-readable concept name |
| `alpha` | float | Cosine similarity to parent direction. Use `0.0` for root nodes. |
| `beta` | float | Orthogonal mixing weight. Typically `sqrt(1-α²)` but can be set independently for ablations. |
| `mutually_exclusive_children` | bool | Whether children are mutually exclusive siblings. |
| `children` | array | Nested child nodes. Empty list `[]` for leaf nodes. |

`alpha` and `beta` need not satisfy `α²+β²=1` strictly — the initializer L2-normalizes
the resulting vector. Setting them independently allows controlled ablations of the
orthogonality assumption.

---

## Files to Modify

### 1. `sae_lens/synthetic/hierarchy/node.py` — Add `alpha`, `beta`, and `label` fields

`HierarchyNode` currently has no per-edge semantic similarity or concept name. Add:

```python
@dataclass
class HierarchyNode:
    feature_index: int | None
    children: Sequence["HierarchyNode"]
    mutually_exclusive_children: bool
    scale_children_by_parent: bool
    feature_id: str | None = None
    label: str | None = None    # NEW: human-readable concept name (e.g. "Deceptive Reasoning")
    alpha: float = 0.0          # NEW: semantic similarity to parent (0.0 for roots)
    beta: float = 1.0           # NEW: orthogonal mixing weight (1.0 for roots)
```

`alpha` and `beta` on each node store the mixing coefficients relative to its parent.
Roots have `alpha=0.0, beta=1.0` (pure random direction).

Also update `from_dict()` / `to_dict()` to round-trip these fields.

---

### 2. `sae_lens/synthetic/hierarchy/config.py` — Add JSON path

`HierarchyConfig` controls tree structure but not feature vector geometry. Add a single
new field:

```python
@dataclass
class HierarchyConfig:
    total_root_nodes: int
    branching_factor: int | tuple[int, int]
    max_depth: int
    mutually_exclusive_portion: float
    mutually_exclusive_min_depth: int
    mutually_exclusive_max_depth: int
    compensate_probabilities: bool
    scale_children_by_parent: bool
    # NEW:
    semantic_dictionary_path: str | None = None  # path to JSON; if set, overrides random generation
```

When `semantic_dictionary_path` is set, `generate_hierarchy()` reads the JSON and ignores
`total_root_nodes`, `branching_factor`, and `max_depth` (the JSON defines the topology).
All other config fields (`mutually_exclusive_portion`, `compensate_probabilities`,
`scale_children_by_parent`) still apply and are read from the config, not the JSON.

---

### 3. NEW FILE: `sae_lens/synthetic/semantic_labels.py` — JSON loader

This file owns all JSON parsing logic and the `ConceptNode` intermediate representation:

```python
@dataclass
class ConceptNode:
    label: str
    alpha: float
    beta: float
    mutually_exclusive_children: bool
    children: list["ConceptNode"]

def load_semantic_dictionary(path: str) -> list[ConceptNode]:
    """Load a list of root ConceptNodes from a JSON file.
    Validates required fields and raises ValueError for missing keys or cycles."""

def concept_node_to_hierarchy_node(
    node: ConceptNode,
    feature_index_start: int,
    scale_children_by_parent: bool = False,
) -> tuple[HierarchyNode, int]:
    """Recursively convert ConceptNode tree to HierarchyNode tree.
    Assigns sequential feature indices. Returns (root_node, next_free_index)."""
```

`load_semantic_dictionary()` is the only place that touches the JSON file.

---

### 4. `sae_lens/synthetic/hierarchy/hierarchy.py` — Branch on JSON in `generate_hierarchy()`

`generate_hierarchy()` currently always generates a random forest. Add a branch at the top:

```python
def generate_hierarchy(
    num_features: int,
    config: HierarchyConfig,
    seed: int | None = None,
) -> Hierarchy:
    if config.semantic_dictionary_path is not None:
        concept_roots = load_semantic_dictionary(config.semantic_dictionary_path)
        roots: list[HierarchyNode] = []
        next_index = 0
        for concept_root in concept_roots:
            node, next_index = concept_node_to_hierarchy_node(
                concept_root, next_index, config.scale_children_by_parent
            )
            roots.append(node)
        if next_index > num_features:
            raise ValueError(
                f"JSON defines {next_index} nodes but num_features={num_features}"
            )
        # Features next_index..num_features-1 are free (no hierarchy node assigned)
    else:
        roots = _generate_random_hierarchy(num_features, config, seed)
    modifier = hierarchy_modifier(roots) if roots else None
    return Hierarchy(roots=roots, modifier=modifier)
```

The `num_features` argument is still needed to validate that the JSON does not overflow
the dictionary and to size the `feature_vectors` tensor in the initializer.

No changes to `modifier.py` are needed.

---

### 5. `sae_lens/synthetic/feature_dictionary.py` — Add semantic initializer

Currently `orthogonal_initializer()` makes all feature vectors orthogonal. A new
`semantic_initializer()` reads `alpha` and `beta` from each `HierarchyNode` to set
feature vectors for the hierarchical portion, then orthogonalizes the remainder:

```python
def semantic_initializer(
    hierarchy: Hierarchy,
    num_features: int,
    hidden_dim: int,
) -> FeatureDictionary:
    """
    Initialize feature vectors encoding parent-child similarity via:
        d_child = α · d_parent + β · d_⊥,  then L2-normalize.
    α and β are read from node.alpha and node.beta (populated from JSON).
    Root nodes get random unit vectors.
    Features with indices beyond the hierarchy are orthogonalized freely.
    """
```

Steps:
1. Assign random unit vectors to all root nodes.
2. BFS over each tree: for each child node, compute
   `d_child = node.alpha * d_parent + node.beta * d_⊥` where `d_⊥` is sampled
   orthogonal to `d_parent` via Gram-Schmidt, then L2-normalize.
3. Collect all vectors in feature-index order.
4. Fill remaining indices (`hierarchy.feature_indices_used` to `num_features-1`) with
   random vectors and orthogonalize them using `orthogonalize_embeddings()`.
5. Return `FeatureDictionary` with `feature_vectors` of shape `(num_features, hidden_dim)`.

---

### 6. `sae_lens/synthetic/synthetic_model.py` — Wire up the new initializer

`SyntheticModel.__init__` currently calls `orthogonal_initializer()` when
`OrthogonalizationConfig` is present. Add a branch:

```python
if config.semantic_geometry and config.hierarchy:
    feature_dict = semantic_initializer(hierarchy, num_features, hidden_dim)
elif config.orthogonalization:
    feature_dict = orthogonal_initializer(...)
else:
    feature_dict = FeatureDictionary(...)
```

Add `semantic_geometry: bool = False` to `SyntheticModelConfig`. When
`config.hierarchy.semantic_dictionary_path` is set, `semantic_geometry` should
default to `True` (the JSON defines both topology and geometry).

---

## Data Flow Summary

```
semantic_dictionary.json   (N trees, any number up to num_features nodes total)
        │
        ▼
semantic_labels.load_semantic_dictionary()
        │  list[ConceptNode]  (label, alpha, beta, children)
        ▼
hierarchy.generate_hierarchy()
        │  assigns feature indices 0..N_hierarchical-1 to JSON nodes
        │  features N_hierarchical..num_features-1 → free (no HierarchyNode)
        ▼
feature_dictionary.semantic_initializer()
        │  indices 0..N_hierarchical-1 → d_child = α·d_parent + β·d_⊥
        │  indices N_hierarchical..num_features-1 → orthogonalize_embeddings()
        ▼
FeatureDictionary  (feature_vectors shape: [num_features, hidden_dim])
        │
        ▼
SyntheticModel.sample()  →  training data
```

---

## Summary of Changes by File

| File | Change |
|------|--------|
| `hierarchy/node.py` | Add `alpha: float`, `beta: float`, `label: str \| None`; update `from_dict`/`to_dict` |
| `hierarchy/config.py` | Add `semantic_dictionary_path: str \| None` |
| `hierarchy/hierarchy.py` | Branch on `semantic_dictionary_path` in `generate_hierarchy()`; free features require no extra code |
| `feature_dictionary.py` | Add `semantic_initializer()` — geometric init for hierarchy nodes, orthogonal init for free features |
| `synthetic_model.py` | Add `semantic_geometry: bool` to `SyntheticModelConfig`; branch to `semantic_initializer` |
| `semantic_labels.py` | **New file**: `ConceptNode`, `load_semantic_dictionary()`, `concept_node_to_hierarchy_node()` |

No changes are needed to `modifier.py` (activation constraints already correct),
`activation_generator.py`, `evals.py`, or `training.py`.
