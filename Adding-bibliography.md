Two steps, both needed:

---

**1. Add the entry to `references.bib`**

The easiest way is by arXiv ID using the Zotero MCP (which I already set up):

> "Add arXiv:2501.12345 to my Zotero library"

Then export the BibTeX entry and paste it into `references.bib`. Or add it manually following the existing format:

```bibtex
@misc{lastname_keyword_year,
    title  = {Full Paper Title},
    author = {Lastname, Firstname and Lastname2, Firstname2},
    year   = {2025},
    url    = {https://arxiv.org/abs/2501.12345},
    doi    = {10.48550/arXiv.2501.12345},
    note   = {arXiv:2501.12345},
}
```

The cite key (`lastname_keyword_year`) is what you'll use in the text.

---

**2. Cite it in `Krampis-SyntheticLLMs-2026.md`**

Use pandoc's `[@citekey]` syntax anywhere in the prose:

```markdown
Recent work on feature geometry [@lastname_keyword_year] shows that...

Multiple works have addressed this [@karvonen_saebench_2025; @gurnee_finding_2023].
```

Pandoc/citeproc replaces these with `[1]`, `[2]`, etc. and auto-generates the References section.

---

**3. Push — CI does the rest**

```bash
git add references.bib Krampis-SyntheticLLMs-2026.md
git commit -m "refs: add Smith 2025"
git push origin main
```

The CI workflow detects changes to `references.bib` or the markdown, runs `pandoc --citeproc --bibliography references.bib --csl ieee.csl`, and commits the regenerated `index.html` and PDF back to main automatically.

---

**Citation variants:**

| Syntax                     | Output                                                      |
| -------------------------- | ----------------------------------------------------------- |
| `[@smith2025]`             | `[1]`                                                       |
| `[@smith2025; @jones2024]` | `[1], [2]`                                                  |
| `[-@smith2025]`            | suppress author, show year (less relevant for IEEE numeric) |
