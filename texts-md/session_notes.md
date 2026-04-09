# Session notes: geometric feature invariance in SAEs
### Based on Krampis (March 2026) — Section 3.3

---

## Context

These notes cover a Q&A session explaining the mathematical foundations behind **section 3.3** of the paper: *Compositional Feature Directions with Hierarchical Constraints*. The core idea is that child feature directions are constructed to be geometrically dependent on their parent directions, creating a controlled cosine similarity equal to α.

The child direction formula is:

$$\mathbf{d}_{child} = \alpha \cdot \mathbf{d}_{parent} + \beta \cdot \mathbf{d}_{\perp}$$

---

## Q1 — What does it mean for a vector to be L2-normalized?

A vector is scaled so that its length (Euclidean norm) equals exactly 1:

$$\|\mathbf{d}\|_2 = \sqrt{\sum_i d_i^2} = 1$$

This makes it a **unit vector** — it encodes only *direction*, not magnitude.

---

## Q2 — Why is the child vector transposed in the dot product?

The dot product $\mathbf{d}_{child}^T \mathbf{d}_{parent}$ requires multiplying a **row vector** by a **column vector**. The transpose turns $\mathbf{d}_{child}$ from a column into a row so the dimensions align for matrix multiplication, yielding a scalar.

---

## Q3 — Why does the dot product of two unit vectors equal their cosine similarity?

The general formula for cosine similarity is:

$$\cos(\theta) = \frac{\mathbf{a}^T \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|}$$

When both vectors are L2-normalized (unit-length), each denominator term equals 1, so the formula collapses to just the dot product:

$$\cos(\theta) = \mathbf{a}^T \mathbf{b}$$

---

## Q4 — Why does setting α > 0 create cosine similarity equal to α?

After normalizing $\mathbf{d}_{child}$, its dot product with $\mathbf{d}_{parent}$ equals exactly α. Here's why:

**Before normalization**, the dot product of the unnormalized child with the parent is:

$$(\alpha \cdot \mathbf{d}_{parent} + \beta \cdot \mathbf{d}_\perp)^T \mathbf{d}_{parent} = \alpha \underbrace{(\mathbf{d}_{parent}^T \mathbf{d}_{parent})}_{=1} + \beta \underbrace{(\mathbf{d}_\perp^T \mathbf{d}_{parent})}_{=0} = \alpha$$

The β term vanishes because $\mathbf{d}_\perp$ is orthogonal to $\mathbf{d}_{parent}$.  
After normalization, the cosine similarity is exactly α.

**Interpretation of α:**

| α value | Geometric meaning | Semantic interpretation |
|---|---|---|
| α = 0 | Child ⊥ Parent | No inherited meaning; fully independent |
| α → 1 | Child ≈ Parent | Almost the same concept |
| 0 < α < 1 | Partial alignment | Child inherits part of parent's meaning, plus its own unique component |

---

## Q5 — How does Gram-Schmidt orthogonalization work to produce d_perp?

$$\mathbf{d}_{\perp} = \mathbf{v} - (\mathbf{v} \cdot \mathbf{d}_{parent})\, \mathbf{d}_{parent}$$

The formula does one thing: **strip away whatever part of v points in the parent's direction, keeping only the leftover part that is purely sideways to it.**

**Step by step:**

1. Start with any random vector **v** (which generally points partly toward the parent and partly sideways).
2. Compute the **projection** of v onto the parent direction: $(\mathbf{v} \cdot \mathbf{d}_{parent})\, \mathbf{d}_{parent}$. The scalar $\mathbf{v} \cdot \mathbf{d}_{parent}$ measures how much of v points toward the parent; multiplying back by $\mathbf{d}_{parent}$ turns it into a vector — the "shadow" of v cast onto the parent axis.
3. **Subtract** that projection from v. What remains is the component of v that was never pointing toward the parent — purely perpendicular.

**Proof that d_perp is orthogonal to d_parent:**

$$\mathbf{d}_\perp \cdot \mathbf{d}_{parent} = [\mathbf{v} - (\mathbf{v} \cdot \mathbf{d}_{parent})\mathbf{d}_{parent}] \cdot \mathbf{d}_{parent} = \underbrace{\mathbf{v} \cdot \mathbf{d}_{parent}}_{s} - s \underbrace{\mathbf{d}_{parent} \cdot \mathbf{d}_{parent}}_{=1} = s - s = 0$$

The result $\mathbf{d}_\perp$ represents the **pure specialization direction** for the child — the component that captures what the child concept adds *beyond* the parent.

---

## Q6 — Is the dot product of a vector with itself equal to 1?

**Only if the vector is unit-length (L2-normalized).** For a unit vector:

$$\mathbf{d} \cdot \mathbf{d} = \sum_i d_i^2 = \|\mathbf{d}\|^2 = 1$$

For a general (non-normalized) vector, $\mathbf{d} \cdot \mathbf{d} = \|\mathbf{d}\|^2$, which equals the squared length — not necessarily 1.

---

## Summary

The key insight of section 3.3 is that α acts as a **direct geometric dial** for semantic relatedness. By construction:

- α is embedded into the child vector via the mixing formula.
- The orthogonal component $\mathbf{d}_\perp$ (obtained via Gram-Schmidt) ensures the β term contributes zero to the dot product with the parent.
- After normalization, $\cos(\theta) = \mathbf{d}_{child}^T \mathbf{d}_{parent} = \alpha$ exactly.

This creates testable predictions for SAE evaluation: well-functioning SAEs should recover decoder directions where child features have cosine similarity ≈ α with parent features, and ablating parent latents should impair child feature reconstruction more than unrelated features.
