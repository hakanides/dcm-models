# DCM Pipeline Issues

This directory contains GitHub-ready issue templates for the DCM/HCM/MXL pipeline.

## Issue Summary

| # | Title | Priority | Category | Labels |
|---|-------|----------|----------|--------|
| 01 | `final_comparison.py` path mismatch | High | Pipeline | `bug`, `pipeline` |
| 02 | `exempt` variables constant by alternative | High | Data/Convergence | `bug`, `identification` |
| 03 | Extreme choice imbalance (~91% alt3) | High | Data/Convergence | `bug`, `data-generation` |
| 04 | Fee magnitudes cause numerical instability | High | Data/Convergence | `bug`, `numerical-stability` |
| 05 | Status quo alternative fully constant | Medium | Data/Design | `enhancement`, `experimental-design` |
| 06 | MXL insufficient simulation draws | Medium | Convergence | `bug`, `MXL` |
| 07 | LR test produces negative values | High | Statistics | `bug`, `correctness` |
| 08 | Models lack explicit names | Low | Usability | `enhancement`, `usability` |
| 09 | Add data QA checks | High | Developer Experience | `enhancement`, `data-validation` |
| 10 | Provide `run_all.sh` script | Low | Usability | `enhancement`, `usability` |

## Recommended Order of Resolution

### Phase 1: Fix Data Generation (blocks everything else)
1. **Issue #03** - Choice imbalance (root cause of many problems)
2. **Issue #04** - Fee scaling (numerical stability)
3. **Issue #02** - Exempt collinearity (identification)
4. **Issue #05** - Status quo design (related to #03)

### Phase 2: Add Validation & Safety
5. **Issue #09** - Data QA checks (prevent future issues)
6. **Issue #07** - LR test correctness (statistical validity)

### Phase 3: Pipeline Usability
7. **Issue #01** - Path configuration (unblock users)
8. **Issue #06** - MXL draws (estimation quality)
9. **Issue #10** - Run script (reproducibility)
10. **Issue #08** - Model naming (traceability)

## Quick Copy Commands

To create these as GitHub issues using `gh` CLI:

```bash
cd issues/

# Create all issues (adjust labels as needed)
gh issue create --title "final_comparison.py cannot find model outputs" --body-file issue_01_final_comparison_paths.md --label "bug,pipeline"

gh issue create --title "exempt variables constant by alternative causes non-identification" --body-file issue_02_exempt_collinearity.md --label "bug,data-generation,convergence"

gh issue create --title "Extreme choice imbalance (91% alt3) causes quasi-separation" --body-file issue_03_choice_imbalance.md --label "bug,data-generation,high-priority"

gh issue create --title "Fee magnitudes cause exp overflow/underflow" --body-file issue_04_fee_scaling.md --label "bug,numerical-stability,high-priority"

gh issue create --title "Status quo alternative has constant attributes" --body-file issue_05_status_quo_constant.md --label "enhancement,experimental-design"

gh issue create --title "MXL uses insufficient simulation draws" --body-file issue_06_mxl_draws.md --label "bug,convergence"

gh issue create --title "LR test produces negative values (incorrect sign)" --body-file issue_07_lr_test_sign.md --label "bug,statistics,high-priority"

gh issue create --title "Models lack explicit names" --body-file issue_08_model_naming.md --label "enhancement,usability"

gh issue create --title "Add comprehensive data QA checks" --body-file issue_09_data_qa_checks.md --label "enhancement,data-validation,high-priority"

gh issue create --title "Provide run_all.sh script" --body-file issue_10_run_script.md --label "enhancement,usability"
```

## Success Criteria (from original feedback)

### Goal A: Reproducible Pipeline
- [ ] `final_comparison.py` finds outputs without manual symlinks
- [ ] Single command runs everything

### Goal B: Estimation Convergence
- [ ] Biogeme converges for core models without warnings
- [ ] Convergence status saved in results
- [ ] LR tests only on converged, nested pairs

### Goal C: Identifiable Synthetic Data
- [ ] No constant-by-alternative regressors estimated with ASCs
- [ ] No alternative with >80% choice share
- [ ] Utility differences in moderate range

### Goal D: Interpretation-Ready Outputs
- [ ] Stable estimates across reruns
- [ ] Model rankings based on converged models only
