# Short Communication Experimental Scope

## Working Title

Physics-Guided AI for Adaptive Photon Allocation in Defense-Oriented
Single-Photon LiDAR

## Central Research Question

Can a physics-guided AI allocation policy improve target-detection reliability
while reducing transmitted photon use compared with fixed and heuristic
single-photon LiDAR allocation methods?

## Baselines

1. Uniform allocation
2. Greedy posterior allocation
3. Posterior-entropy allocation
4. Approximate information-gain allocation

The random policy may remain in the repository for software validation but
will not be emphasized in the manuscript.

## Proposed Method

One lightweight physics-guided neural allocation policy.

The policy will use:

- current posterior probabilities;
- measurements already assigned to each region;
- remaining photon budget;
- predicted signal-to-background ratio;
- target-range information;
- atmospheric transmission estimates.

## Evaluation Ranges

- 500 m
- 1500 m
- 3000 m

## Atmospheric Conditions

- Clear air
- Fog or smoke degraded channel

Only one degraded condition will be used in the principal paper.

## Photon Budgets

Evaluate a compact sweep of four total scene budgets:

- 1.0e12 photons
- 2.0e12 photons
- 3.5e12 photons
- 5.0e12 photons

## Random Seeds

Use five independent seeds for the pilot and short communication:

- 11
- 22
- 33
- 44
- 55

## Target Conditions

For each principal configuration:

- target-present episodes at the three evaluated ranges;
- a small target-absent set for false-alarm estimation.

## Primary Metrics

1. Detection rate
2. Mean photons required for detection
3. Mean sensing steps
4. False-alarm rate

## Secondary Metrics

- final target posterior;
- photon savings relative to uniform allocation;
- inference latency.

## Minimum Paper Outputs

### Figures

1. Proposed optical sensing and AI allocation architecture.
2. Detection rate versus photon budget.
3. Mean photons to detection versus range.
4. Clear-air versus degraded-atmosphere comparison.

### Tables

1. Optical, detector, atmospheric and training parameters.
2. Baseline and proposed-method comparison.
3. Compact ablation study.

## Ablation

Compare the proposed AI policy with:

- full physics-guided features;
- physics features removed;
- fixed uniform allocation.

## Excluded From This Short Communication

- five different atmospheric classes;
- large-scale model mismatch analysis;
- moving-target tracking;
- multi-target scenes;
- hardware field deployment;
- quantum advantage claims;
- more than one neural architecture;
- extensive hyperparameter optimization;
- large 20–30 seed studies.

These may be reserved for a future full-length article.
