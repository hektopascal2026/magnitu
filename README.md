# Magnitu

Magnitu is a machine-learning relevance engine for Seismo.  
It learns your labeling decisions (`investigation_lead`, `important`, `background`, `noise`) and pushes scores + a lightweight recipe back to Seismo for live ranking.

## Current Stack (Magnitu 2)

- **Local model**: transformer embeddings (`xlm-roberta-base` by default) + MLP classifier
- **Seismo runtime**: keyword recipe evaluated in PHP (distilled from local model)
- **Data sync**: pull entries/labels from Seismo, push scores/recipe/labels back

## Key Features

- **Main labeling page**
  - Smart queue (`uncertain`, `conflict`, `diverse`, `new`)
  - Source filter (`All`, `Legislation`, `News`)
  - Keyboard shortcuts for fast labeling
- **Magnitu Mini**
  - Mobile-friendly fast labeling UI
  - Source filter (`All`, `Legislation`, `News`)
  - Reliable retry queue for failed label pushes
- **Sync modes**
  - **Sync**: quick incremental pull
  - **Full Sync**: source-by-source backfill + repeated embedding passes for coverage
- **Progress tracking**
  - Live progress bars for Sync / Full Sync / Push in main UI
  - Background jobs with status polling
- **Phrase-aware recipe distillation**
  - Uses unigrams, bigrams, trigrams
  - Exports both strong **positive** and **negative** features per class
  - Includes legal-template priors and reasoning-based phrase boosts
- **Explainability**
  - Transformer explanations include matched phrase-level signals from active recipe
  - Dashboard shows learned legal phrase patterns (impact vs procedural/noise)

## Typical Workflow

1. **Sync** (or **Full Sync** when you need full coverage)
2. **Label** entries (main app or Mini)
3. **Train**
4. **Push to Seismo**
5. Review top-ranked entries and correct mistakes

## Run Locally

```bash
git clone https://github.com/hektopascal2026/magnitu.git
cd magnitu
bash install/bootstrap.sh
./start.sh
```

Open: `http://127.0.0.1:8000`

## Notes

- Seismo currently uses one active recipe at a time. Last push wins.
- If you see Seismo recipe/full-model mismatch, run **Full Sync**, then **Train**, then **Push**.
- Python 3.9 compatibility is preserved.
