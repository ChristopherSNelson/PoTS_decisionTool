# CLAUDE.md - Biotech Portfolio PoTS Decision Engine

> Adapted from Boris Cherny's Claude Code setup for a biopharma portfolio decision tool.
>
> **For Claude**: Sections marked *[AGENT RULES]* are directives you must follow.
> **For the human**: Sections marked *[OPERATOR NOTES]* are reminders for the user's workflow.
> **Shared**: Unmarked sections apply to both.
>
> **Self-updating rule** *[AGENT RULES]*: After any correction or mistake, propose a specific
> addition to the Mistakes log at the bottom of this file. The user will end corrections with:
> "Now update CLAUDE.md so you don't make that mistake again."

---

## Environment constraints *[AGENT RULES]*

- **Machine**: Apple M1 Pro, 16 GB unified memory, macOS.
- **Python**: 3.11+. Core stack: numpy, pandas, scipy, matplotlib, ipywidgets.
- **Notebook execution**: Use `jupyter nbconvert --to notebook --execute` for headless runs. Always verify plots were generated after execution.
- **ARM/Apple Silicon awareness**: Default to ARM-native wheels. Flag Rosetta fallbacks explicitly.

---

## Project structure *[AGENT RULES]*

```
PoTS_decisionTool/
  portfolio_demo.ipynb    # Main notebook - all analysis and interactive controls
  data/
    assets.csv            # Portfolio definition (swap this to change portfolios)
    repurpose_map.csv     # Repurpose candidates with similarity scores
  plots/                  # All generated plots (saved by notebook cells)
  README.md               # Overview with embedded plot images
```

- **Data lives in CSV, logic lives in the notebook.** Never hardcode asset parameters in code cells.
- **All plots must be saved to `plots/`** and should be visible in the README.
- After modifying the notebook, always re-execute and verify plots before reporting success.

---

## Domain rules - Biopharma portfolio decisions *[AGENT RULES]*

- **PoTS (probability of technical success)** is modeled as a Beta-Binomial posterior. Prior parameters come from `data/assets.csv`.
- **Decision thresholds are phase-dependent** - the bar rises with capital commitment:

  | Phase     | Kill (mean <) | Go (mean >) |
  |-----------|---------------|-------------|
  | Phase I   | 0.10          | 0.15        |
  | Phase II  | 0.10          | 0.25        |
  | Phase III | 0.15          | 0.40        |

- **Kill uses posterior mean**, not CI upper. CI-upper-based kill is too conservative with typical interim data sizes - it almost never fires. Mean-based kill is simpler and visually intuitive. The 90% CI still displays on charts for uncertainty communication.
- **Correlation propagation**: When an asset fails, correlated assets receive virtual failures weighted by the correlation matrix. This is the core of portfolio-level thinking.
- **Repurposing is biology-driven**, not an arbitrary middle bucket. Repurpose PoTS = original PoTS * similarity score. Candidates come from `data/repurpose_map.csv`.
- **Expected NPV**: `eNPV = PoTS * Revenue - Cost`. Capital allocation is greedy by eNPV rank within a fixed budget.
- **Monte Carlo**: Gaussian copula for correlated sampling from Beta posteriors. Report median (not mean) as the headline portfolio value - the distribution is skewed.
- **All data is simulated.** This is a toy model. Don't claim clinical validity.

---

## Visualization rules *[AGENT RULES]*

- **matplotlib dollar signs**: `$` triggers mathtext in legends/labels. Use `chr(36)` or `set_parse_math(False)` on legend text to avoid mangling.
- **Money formatting**: Use `$` prefix. Values >= 1000M display as `$X.YB`. Always include spaces around "to" in ranges.
- **Bar chart labels**: Place labels inside bars (white bold) when the bar is large enough, outside (left-aligned) when small. Never let labels extend beyond the plot boundary.
- **Error bars on probabilities**: These are 90% credible intervals from the Beta posterior, not arbitrary. Clip at 0 - probabilities can't be negative.
- **Phase labels**: Include phase in x-axis labels on decision charts so reviewers see why thresholds differ.

---

## Coding standards *[AGENT RULES]*

- **Python**: Type hints on function signatures. Docstrings on non-trivial functions. Keep notebook cells focused - one concept per cell.
- **No em dashes** (--). Use regular hyphens (-) or rewrite.
- **Notebook cells**: Markdown headers before each code section. Print key dataframes so outputs are visible in the executed notebook.
- **Git**: Conventional commits. Feature branches off `main`.

---

## Verification loops *[AGENT RULES]*

After modifying the notebook:

1. Run `jupyter nbconvert --to notebook --execute portfolio_demo.ipynb --output portfolio_demo.ipynb`
2. Verify all plots in `plots/` were regenerated (check file timestamps or read the images)
3. If a plot has layout issues (overlapping labels, clipped text), fix and re-run before reporting success

Do not report a task as complete unless the notebook executes cleanly and plots look correct.

---

## Workflow philosophy *[AGENT RULES]*

- **Plan before executing**: For complex changes, outline the approach first.
- **When things go sideways**: Stop, reassess, re-plan. Do not keep pushing a broken approach.
- **Self-correcting CLAUDE.md**: After any correction or mistake, propose an update to this file.
- **Handoff discipline**: When context is getting long (15+ turns), proactively offer to write a `HANDOFF.md`.

---

## Biopharma context *[AGENT RULES]*

- Target audience: fellowship reviewers, investment committees, portfolio managers.
- Documentation should be clear enough for a non-quant to understand the decision logic, and precise enough for a statistician to critique the methodology.
- When discussing therapeutic areas, indications, or mechanisms - be precise. Don't conflate SLE with lupus nephritis, or NAFLD with NASH.
- Phase transition benchmarks (BIO/QLS/Informa): Phase I-II ~55%, Phase II-III ~30%, Phase III-approval ~55%, overall ~10%.

---

## Usage economy *[AGENT RULES]*

- **Keep outputs compact**: Don't dump large stdout. Limit data previews.
- **Don't re-read files unnecessarily**: If you wrote it recently, reference by name.
- **Batch work**: Multiple related changes in one turn, not spread across rounds.

---

## Mistakes log

<!-- Add entries here as they happen. Format: date, what went wrong, rule added. -->
- 2026-03-28: matplotlib `$` in legend labels triggers mathtext, collapsing spaces. Rule: use `set_parse_math(False)` or `chr(36)` for dollar signs in legend text.
- 2026-03-28: Bar chart labels placed outside large bars get clipped at plot boundary. Rule: place labels inside bars (white bold) when bar is large enough.
- 2026-03-28: Used flat kill/go thresholds across all phases. Rule: thresholds must be phase-dependent.

---

## Project-specific overrides

- This project uses Jupyter notebooks, not scripts or pipelines.
- No patient data. All data is synthetic.
- Interactive controls via ipywidgets (not Streamlit).
- Rare disease / biopharma focus for UK fellowship context.

---

## Future directions

### Streamlit deployment (not attempted - blocked)

`app.py` and `engine.py` exist and the Streamlit app runs correctly locally (`streamlit run app.py`). Deployment to Streamlit Community Cloud was attempted but failed repeatedly with "The app's code is not connected to a remote GitHub repository" despite the repo being public and credentials being correct. Root cause unclear - likely a GitHub App authorization issue on the Streamlit side that requires manual intervention through their support.

**If revisiting**: Try Hugging Face Spaces (huggingface.co - create Space, type Streamlit, push same code). Simpler auth than Streamlit Cloud. Alternatively, the notebook is publicly viewable at `https://nbviewer.org/github/ChristopherSNelson/PoTS_decisionTool/blob/main/portfolio_demo.ipynb` which is sufficient for fellowship review purposes.

**Do not attempt Streamlit Community Cloud again without a clear fix in hand.**
