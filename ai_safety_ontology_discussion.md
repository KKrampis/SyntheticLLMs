# AI Safety Ontology — Semantic Dictionary Discussion

> Exported from a Claude conversation on building an AI safety concept ontology using the **SynthSAEBench / SAELens semantic dictionary** format (semantic_dictionary.json).

---

## Background: Semantic Dictionary Format

A semantic dictionary encodes parent-child concept relationships directly into the geometry of feature vectors. Instead of initialising feature vectors to be as orthogonal as possible, each child feature direction is constructed as a blend of its parent's direction and a new orthogonal component:

```
d_child = α · d_parent + β · d_⊥
```

- **α** (semantic similarity coefficient): cosine similarity between child and parent directions after normalisation. Higher values mean the child is geometrically closer to its parent.
- **β** (orthogonal mixing weight): controls how much of the child's direction is unique. Typically `β = sqrt(1 - α²)`, which ensures `cos(θ_child, parent) = α` exactly.
- **d_⊥**: a unit vector orthogonal to the parent direction, derived via Gram-Schmidt.

Root nodes keep random unit vectors. Children only fire when their parent is active, and siblings can be mutually exclusive.

---

## Where to Get Terms and Knowledge

### Existing AI Safety Ontologies & Taxonomies

- **AI Safety Levels / AISES textbook** (aisafety.info) — covers risks, threat models, alignment concepts
- **MITRE ATLAS** — adversarial ML taxonomy, machine-readable and well-structured
- **AI Incident Database (AIID)** — real-world failure modes, good for grounding abstract concepts
- **Anthropic, DeepMind, ARC, MIRI papers** — primary sources for alignment-specific terminology (deceptive alignment, mesa-optimization, inner/outer alignment, etc.)
- **AI Risk Repository (MIT)** — recently published (~2024), explicitly structured as a taxonomy with ~700 risks across multiple levels

### Crowdsourced / Community

- **LessWrong & Alignment Forum tags** — the tag hierarchy there is itself a rough ontology
- **AGI Safety Fundamentals curriculum** — well-curated concept list with dependencies

### Semi-automated Approach

Use an LLM to extract terms from a corpus of papers (e.g., MIRI's research agenda, Anthropic's core views doc), then have domain experts validate the tree structure.

### Recommended Workflow for a Full Ontology

1. Export the **MIT AI Risk Repository** taxonomy (machine-readable, ~700 terms, already hierarchically structured)
2. Combine with **Alignment Forum tag taxonomy** for more research-facing concepts
3. Map top-level clusters to root nodes in JSON
4. Use an LLM to suggest alpha values based on semantic relatedness
5. Human review of the resulting tree

---

## Design Principles for Alpha / Beta Assignment

| Principle | Guidance |
|-----------|----------|
| Root nodes | α=0.0, β=1.0 always |
| Higher α | Child is geometrically (and semantically) closer to parent |
| Distinct siblings | Give siblings different α values so their vectors don't collapse |
| β formula | `β = sqrt(1 - α²)` preserves unit norm; can be set independently for ablations |
| Mutually exclusive children | Set `true` at root level; usually `false` lower down where sub-types co-occur |

---

## Example Tree 1 — Misalignment

### Diagram

```
Misalignment  (root, α=0.0, β=1.0)
├── Inner Alignment Failure  (α=0.6, β=0.8)
│   ├── Mesa-optimizer Misalignment  (α=0.65, β=0.76)
│   └── Proxy Gaming  (α=0.5, β=0.866)
└── Outer Alignment Failure  (α=0.5, β=0.866)
    ├── Reward Hacking  (α=0.6, β=0.8)
    └── Goal Misgeneralization  (α=0.55, β=0.835)
```

### Visual (inline SVG)

```svg
<svg width="100%" viewBox="0 0 680 310" role="img"
     xmlns="http://www.w3.org/2000/svg">
  <title>Misalignment ontology tree</title>
  <desc>Root: Misalignment. Children: Inner Alignment Failure and Outer Alignment Failure. Leaves: Mesa-optimizer Misalignment, Proxy Gaming, Reward Hacking, Goal Misgeneralization.</desc>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
            stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <!-- Root -->
  <rect x="220" y="20" width="240" height="48" rx="8"
        fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
  <text x="340" y="40" text-anchor="middle" font-size="14"
        font-weight="500" fill="#3C3489">Misalignment</text>
  <text x="340" y="58" text-anchor="middle" font-size="12"
        fill="#534AB7">α=0.0, β=1.0 (root)</text>

  <!-- Connectors L1 -->
  <line x1="290" y1="68" x2="165" y2="115"
        stroke="#888780" stroke-width="0.5" marker-end="url(#arrow)"/>
  <line x1="390" y1="68" x2="515" y2="115"
        stroke="#888780" stroke-width="0.5" marker-end="url(#arrow)"/>

  <!-- Inner Alignment Failure -->
  <rect x="60" y="117" width="210" height="48" rx="8"
        fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
  <text x="165" y="137" text-anchor="middle" font-size="14"
        font-weight="500" fill="#085041">Inner alignment failure</text>
  <text x="165" y="155" text-anchor="middle" font-size="12"
        fill="#0F6E56">α=0.6, β=0.8</text>

  <!-- Outer Alignment Failure -->
  <rect x="410" y="117" width="210" height="48" rx="8"
        fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
  <text x="515" y="137" text-anchor="middle" font-size="14"
        font-weight="500" fill="#085041">Outer alignment failure</text>
  <text x="515" y="155" text-anchor="middle" font-size="12"
        fill="#0F6E56">α=0.5, β=0.866</text>

  <!-- Connectors L2 left -->
  <line x1="120" y1="165" x2="90" y2="215"
        stroke="#888780" stroke-width="0.5" marker-end="url(#arrow)"/>
  <line x1="210" y1="165" x2="230" y2="215"
        stroke="#888780" stroke-width="0.5" marker-end="url(#arrow)"/>

  <!-- Connectors L2 right -->
  <line x1="468" y1="165" x2="450" y2="215"
        stroke="#888780" stroke-width="0.5" marker-end="url(#arrow)"/>
  <line x1="562" y1="165" x2="580" y2="215"
        stroke="#888780" stroke-width="0.5" marker-end="url(#arrow)"/>

  <!-- Leaf: Mesa-optimizer -->
  <rect x="20" y="217" width="130" height="48" rx="8"
        fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
  <text x="85" y="237" text-anchor="middle" font-size="13"
        font-weight="500" fill="#2C2C2A">Mesa-optimizer</text>
  <text x="85" y="254" text-anchor="middle" font-size="11"
        fill="#5F5E5A">misalignment</text>

  <!-- Leaf: Proxy Gaming -->
  <rect x="170" y="217" width="120" height="48" rx="8"
        fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
  <text x="230" y="241" text-anchor="middle" font-size="13"
        font-weight="500" fill="#2C2C2A">Proxy gaming</text>

  <!-- Leaf: Reward Hacking -->
  <rect x="390" y="217" width="120" height="48" rx="8"
        fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
  <text x="450" y="241" text-anchor="middle" font-size="13"
        font-weight="500" fill="#2C2C2A">Reward hacking</text>

  <!-- Leaf: Goal Misgeneralization -->
  <rect x="530" y="217" width="130" height="48" rx="8"
        fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
  <text x="595" y="237" text-anchor="middle" font-size="13"
        font-weight="500" fill="#2C2C2A">Goal</text>
  <text x="595" y="254" text-anchor="middle" font-size="11"
        fill="#5F5E5A">misgeneralization</text>
</svg>
```

---

## Example Tree 2 — Deceptive Behavior

### Diagram

```
Deceptive Behavior  (root, α=0.0, β=1.0)
├── Strategic Deception  (α=0.65, β=0.76)
│   ├── Goal Misrepresentation  (α=0.6, β=0.8)
│   └── Selective Omission  (α=0.5, β=0.866)
└── Sycophancy  (α=0.55, β=0.835)
    ├── Approval Seeking  (α=0.65, β=0.76)
    └── Opinion Mirroring  (α=0.55, β=0.835)
```

### Visual (inline SVG)

```svg
<svg width="100%" viewBox="0 0 680 310" role="img"
     xmlns="http://www.w3.org/2000/svg">
  <title>Deceptive behavior ontology tree</title>
  <desc>Root: Deceptive Behavior. Children: Strategic Deception and Sycophancy. Leaves: Goal Misrepresentation, Selective Omission, Approval Seeking, Opinion Mirroring.</desc>
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
            stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <!-- Root -->
  <rect x="190" y="20" width="300" height="48" rx="8"
        fill="#FAECE7" stroke="#993C1D" stroke-width="0.5"/>
  <text x="340" y="40" text-anchor="middle" font-size="14"
        font-weight="500" fill="#4A1B0C">Deceptive behavior</text>
  <text x="340" y="58" text-anchor="middle" font-size="12"
        fill="#993C1D">α=0.0, β=1.0 (root)</text>

  <!-- Connectors L1 -->
  <line x1="280" y1="68" x2="175" y2="115"
        stroke="#888780" stroke-width="0.5" marker-end="url(#arrow2)"/>
  <line x1="400" y1="68" x2="515" y2="115"
        stroke="#888780" stroke-width="0.5" marker-end="url(#arrow2)"/>

  <!-- Strategic Deception -->
  <rect x="60" y="117" width="230" height="48" rx="8"
        fill="#FBEAF0" stroke="#993556" stroke-width="0.5"/>
  <text x="175" y="137" text-anchor="middle" font-size="14"
        font-weight="500" fill="#4B1528">Strategic deception</text>
  <text x="175" y="155" text-anchor="middle" font-size="12"
        fill="#993556">α=0.65, β=0.76</text>

  <!-- Sycophancy -->
  <rect x="410" y="117" width="210" height="48" rx="8"
        fill="#FBEAF0" stroke="#993556" stroke-width="0.5"/>
  <text x="515" y="137" text-anchor="middle" font-size="14"
        font-weight="500" fill="#4B1528">Sycophancy</text>
  <text x="515" y="155" text-anchor="middle" font-size="12"
        fill="#993556">α=0.55, β=0.835</text>

  <!-- Connectors L2 left -->
  <line x1="128" y1="165" x2="100" y2="215"
        stroke="#888780" stroke-width="0.5" marker-end="url(#arrow2)"/>
  <line x1="222" y1="165" x2="242" y2="215"
        stroke="#888780" stroke-width="0.5" marker-end="url(#arrow2)"/>

  <!-- Connectors L2 right -->
  <line x1="472" y1="165" x2="452" y2="215"
        stroke="#888780" stroke-width="0.5" marker-end="url(#arrow2)"/>
  <line x1="558" y1="165" x2="578" y2="215"
        stroke="#888780" stroke-width="0.5" marker-end="url(#arrow2)"/>

  <!-- Leaf: Goal Misrepresentation -->
  <rect x="22" y="217" width="152" height="48" rx="8"
        fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
  <text x="98" y="237" text-anchor="middle" font-size="13"
        font-weight="500" fill="#2C2C2A">Goal</text>
  <text x="98" y="254" text-anchor="middle" font-size="11"
        fill="#5F5E5A">misrepresentation</text>

  <!-- Leaf: Selective Omission -->
  <rect x="190" y="217" width="140" height="48" rx="8"
        fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
  <text x="260" y="237" text-anchor="middle" font-size="13"
        font-weight="500" fill="#2C2C2A">Selective</text>
  <text x="260" y="254" text-anchor="middle" font-size="11"
        fill="#5F5E5A">omission</text>

  <!-- Leaf: Approval Seeking -->
  <rect x="390" y="217" width="130" height="48" rx="8"
        fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
  <text x="455" y="237" text-anchor="middle" font-size="13"
        font-weight="500" fill="#2C2C2A">Approval</text>
  <text x="455" y="254" text-anchor="middle" font-size="11"
        fill="#5F5E5A">seeking</text>

  <!-- Leaf: Opinion Mirroring -->
  <rect x="538" y="217" width="130" height="48" rx="8"
        fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
  <text x="603" y="237" text-anchor="middle" font-size="13"
        font-weight="500" fill="#2C2C2A">Opinion</text>
  <text x="603" y="254" text-anchor="middle" font-size="11"
        fill="#5F5E5A">mirroring</text>
</svg>
```

---

## Full JSON — Two Example Trees

```json
{
  "trees": [
    {
      "label": "Misalignment",
      "alpha": 0.0,
      "beta": 1.0,
      "mutually_exclusive_children": true,
      "children": [
        {
          "label": "Inner Alignment Failure",
          "alpha": 0.6,
          "beta": 0.8,
          "mutually_exclusive_children": false,
          "children": [
            { "label": "Mesa-optimizer Misalignment", "alpha": 0.65, "beta": 0.76,  "children": [] },
            { "label": "Proxy Gaming",                "alpha": 0.5,  "beta": 0.866, "children": [] }
          ]
        },
        {
          "label": "Outer Alignment Failure",
          "alpha": 0.5,
          "beta": 0.866,
          "mutually_exclusive_children": false,
          "children": [
            { "label": "Reward Hacking",         "alpha": 0.6,  "beta": 0.8,   "children": [] },
            { "label": "Goal Misgeneralization", "alpha": 0.55, "beta": 0.835, "children": [] }
          ]
        }
      ]
    },
    {
      "label": "Deceptive Behavior",
      "alpha": 0.0,
      "beta": 1.0,
      "mutually_exclusive_children": true,
      "children": [
        {
          "label": "Strategic Deception",
          "alpha": 0.65,
          "beta": 0.76,
          "mutually_exclusive_children": false,
          "children": [
            { "label": "Goal Misrepresentation", "alpha": 0.6,  "beta": 0.8,   "children": [] },
            { "label": "Selective Omission",     "alpha": 0.5,  "beta": 0.866, "children": [] }
          ]
        },
        {
          "label": "Sycophancy",
          "alpha": 0.55,
          "beta": 0.835,
          "mutually_exclusive_children": false,
          "children": [
            { "label": "Approval Seeking",  "alpha": 0.65, "beta": 0.76,  "children": [] },
            { "label": "Opinion Mirroring", "alpha": 0.55, "beta": 0.835, "children": [] }
          ]
        }
      ]
    }
  ]
}
```

---

## JSON Node Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `label` | string | Human-readable concept name |
| `alpha` | float | Cosine similarity to parent direction. Use `0.0` for root nodes. |
| `beta` | float | Orthogonal mixing weight. Typically `sqrt(1−α²)`. |
| `mutually_exclusive_children` | bool | If `true`, at most one child fires per sample. Optional, defaults to `false`. |
| `children` | array | Nested child nodes. Empty list `[]` for leaf nodes. |

`alpha` and `beta` need not satisfy `α² + β² = 1` strictly — the initialiser L2-normalises the result. However, when `α² + β² = 1`, the cosine similarity between child and parent exactly equals `α`.

---

## Configuration Example (Python / SAELens)

```python
from sae_lens.synthetic import (
    SyntheticModel,
    SyntheticModelConfig,
    HierarchyConfig,
    ZipfianFiringProbabilityConfig,
    LinearMagnitudeConfig,
    FoldedNormalMagnitudeConfig,
)

cfg = SyntheticModelConfig(
    num_features=16_384,
    hidden_dim=768,

    firing_probability=ZipfianFiringProbabilityConfig(
        exponent=0.5,
        max_prob=0.4,
        min_prob=5e-4,
    ),

    hierarchy=HierarchyConfig(
        semantic_dictionary_path="semantic_dictionary.json",
        compensate_probabilities=True,
        scale_children_by_parent=True,
    ),

    semantic_geometry=True,

    mean_firing_magnitudes=LinearMagnitudeConfig(start=5.0, end=4.0),
    std_firing_magnitudes=FoldedNormalMagnitudeConfig(mean=0.5, std=0.5),

    seed=42,
)

model = SyntheticModel(cfg)
hidden_activations = model.sample(batch_size=1024)
```

---

## Feature Index Assignment

Feature indices are assigned depth-first (root first, then children left to right) for each tree in the `"trees"` array. For example, with `num_features=16384` and a JSON file containing 50 trees of depth 3, branching factor 4 (50 × 85 = 4,250 hierarchical nodes):

```
Features 0 – 4,249      → hierarchical, geometric α/β initialisation
Features 4,250 – 16,383 → free, orthogonalised initialisation
```

A complete 4-ary tree of depth 3 has 1 + 4 + 16 + 64 = **85 nodes** per tree. 128 such trees cover 10,880 hierarchical features, matching the SyntheticLLMs benchmark setup.

---

## Data Flow

```
semantic_dictionary.json   (N trees, any number up to num_features nodes total)
        │
        ▼
load_semantic_dictionary()                # sae_lens.synthetic.semantic_labels
        │  list[ConceptNode]
        ▼
generate_hierarchy()                      # sae_lens.synthetic.hierarchy
        │  features 0..N_hierarchical-1 → HierarchyNode trees with alpha/beta
        │  features N_hierarchical..num_features-1 → free (no HierarchyNode)
        ▼
semantic_initializer()                    # sae_lens.synthetic.feature_dictionary
        │  indices 0..N_hierarchical-1 → d_child = α·d_parent + β·d_⊥
        │  indices N_hierarchical..num_features-1 → orthogonalize_embeddings()
        ▼
FeatureDictionary                         # shape: (num_features, hidden_dim)
        │
        ▼
SyntheticModel.sample()                   # training data for SAE
```

---

## Notes on Design Choices

**Alpha values**: siblings within the same parent are kept distinct (e.g. 0.6 vs 0.5) so their feature vectors don't collapse together geometrically. The higher the alpha, the more the child "inherits" the parent's direction — so "Strategic Deception" (α=0.65) is geometrically closer to "Deceptive Behavior" than "Sycophancy" (α=0.55) is, which feels semantically right.

**Mutually exclusive children**: set to `true` at the root level since a given activation shouldn't simultaneously represent two root concepts. Set to `false` at lower levels where sub-types can co-occur (e.g. a model could exhibit both approval-seeking and opinion-mirroring at once).

**For a full ontology**, the richest sources to start with are the **MIT AI Risk Repository** (machine-readable, ~700 terms, already hierarchically structured) combined with the **Alignment Forum tag taxonomy** for more research-facing concepts.
