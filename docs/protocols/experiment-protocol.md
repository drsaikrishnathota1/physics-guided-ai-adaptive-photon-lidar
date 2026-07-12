# Experiment Protocol

## 1. Optical Forward Model

Photon counts are modeled as time-resolved Poisson random variables:

y[i,k] ~ Poisson(
    signal[i,k] +
    atmospheric_backscatter[i,k] +
    solar_background[i,k] +
    dark_count[i,k]
)

The target-return component depends on:

- allocated transmitted photons;
- target reflectivity;
- range-squared geometric spreading;
- two-way atmospheric transmission;
- receiver aperture;
- optical throughput;
- detector quantum efficiency;
- instrument-response function.

## 2. Atmospheric Conditions

Evaluate at minimum:

- clear air;
- light haze;
- dense haze;
- light fog;
- moderate fog;
- dust;
- smoke.

Use both matched and mismatched extinction/backscatter conditions.

## 3. Detector Effects

Model:

- photon-detection efficiency;
- timing jitter;
- dark-count rate;
- dead time;
- afterpulsing approximation;
- background saturation;
- finite histogram-bin width.

## 4. Target Conditions

Vary:

- range;
- reflectivity;
- target size;
- target-background contrast;
- target motion;
- partial occlusion;
- pose;
- sparse and multiple-target scenes.

## 5. Adaptive Policies

Compare:

1. Uniform photon allocation.
2. Random photon allocation.
3. Greedy maximum-return allocation.
4. Entropy-reduction allocation.
5. Bayesian information-gain allocation.
6. Model-free reinforcement-learning policy.
7. Proposed physics-guided AI policy.
8. Oracle allocation as an upper-bound reference.

## 6. Generalization Tests

Train and test using separated distributions:

- unseen extinction coefficients;
- unseen target reflectivities;
- unseen target ranges;
- unseen backgrounds;
- unseen detector efficiencies;
- unseen combinations of noise sources.

## 7. Ablation Tests

Remove or replace:

- atmospheric transmission input;
- detector model;
- posterior uncertainty;
- physics-based reward term;
- photon-budget constraint;
- recurrent state;
- domain randomization.

## 8. Statistical Design

Use:

- at least five independent training seeds;
- bootstrap 95% confidence intervals;
- paired tests using identical test scenes;
- multiple-comparison correction;
- effect sizes;
- complete reporting of negative results.

## 9. Reproducibility

Each experiment must save:

- Git commit hash;
- configuration file;
- random seeds;
- package versions;
- GPU model;
- training duration;
- raw predictions;
- raw metrics;
- checkpoint identifier.
