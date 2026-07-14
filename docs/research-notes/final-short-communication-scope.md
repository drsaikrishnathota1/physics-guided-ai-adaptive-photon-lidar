# Final Short Communication Scope

## Final Title

Physics-Guided Adaptive Photon Allocation for Defense-Oriented
Single-Photon LiDAR

## Proposed Method

The proposed method is a physics-guided information-gain allocation strategy
that combines:

- posterior target uncertainty;
- predicted signal-to-background ratio;
- measurement-count exploration;
- remaining photon budget;
- range-dependent optical sensing conditions.

The method is not presented as a trained artificial-intelligence model.

## Baselines

1. Uniform allocation
2. Random allocation
3. Greedy posterior allocation
4. Posterior-entropy allocation

## Main Pilot Result

The physics-guided information-gain method achieved the highest target
detection rate in the baseline experiment:

- Information gain: 93.3%
- Greedy: 90.0%
- Entropy: 90.0%
- Uniform: 76.7%
- Random: 63.3%

The information-gain strategy also reduced average photon usage relative
to uniform allocation.

## Scientific Claims

The study demonstrates that combining optical-physics predictions with
posterior uncertainty can improve sequential photon allocation compared
with fixed and simpler heuristic strategies.

## Limitations

- The study is simulation based.
- Thirty episodes were evaluated per policy.
- Every evaluated episode contained a target.
- A dedicated target-absent false-alarm experiment was not performed.
- No neural or reinforcement-learning policy was trained.
- No claim of quantum advantage is made.
- No field-deployment performance is claimed.

## Future Work

Future full-length research may include:

- target-absent false-alarm analysis;
- atmospheric mismatch experiments;
- moving and multiple targets;
- physics-guided neural policies;
- experimental single-photon LiDAR data;
- hardware-in-the-loop validation.
