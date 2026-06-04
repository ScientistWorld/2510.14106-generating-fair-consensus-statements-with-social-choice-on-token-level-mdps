# Evaluation

## Metrics

The core-stability metric is maximum coalition improvement `alpha*`. Values greater than 1 mean some coalition can block the outcome; values at or below 1 satisfy the ex-ante-core constraint.

For single consensus statements, the paper reports Egalitarian Perplexity, the maximum participant-specific perplexity of the generated statement. Lower values mean better worst-participant alignment.

## Baselines and Targets

The paper's synthetic core test expects the fair stochastic outcome to have maximum coalition improvement equal to 1 across polarization levels. The same test reports that uniform and utilitarian alternatives become blockable as polarization increases.

The paper's main text-generation benchmark reports lower EPPL for token-search methods than Best-of-N and Habermas-style baselines on three Habermas Machine scenarios.

## Evaluation Protocol

The included executable reproduction evaluates the synthetic core-stability setting from the paper: multiple random token trees, multiple polarization values, fixed branching factor 3, and depth 4. Evaluation reads saved utility matrices and output distributions, then independently solves the coalition-blocking LP for each instance.
