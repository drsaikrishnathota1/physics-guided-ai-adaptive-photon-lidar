# Physics-Guided AI for Adaptive Photon Allocation in Defense-Oriented Single-Photon LiDAR

## Overview

This repository contains the complete, reproducible research pipeline for developing and
evaluating a physics-guided AI framework that adaptively allocates a limited photon budget
for single-photon LiDAR target detection and ranging.

The central objective is to determine whether an adaptive sensing policy can achieve a
required probability of detection and ranging accuracy using fewer transmitted photons
than fixed or heuristic illumination strategies.

## Target Journal

Optics & Laser Technology — Short Communication

## Research Question

Can a physics-guided adaptive photon-allocation policy reduce the transmitted photon budget
required for reliable target detection and ranging under realistic atmospheric attenuation,
background illumination, detector noise, target-reflectivity variation, and platform motion?

## Primary Contributions

1. A physically grounded single-photon LiDAR simulator based on time-resolved Poisson photon
   arrivals.
2. An adaptive photon-allocation policy informed by posterior uncertainty and optical physics.
3. Evaluation under realistic atmospheric, detector, target, and motion conditions.
4. Comparisons against uniform, random, greedy, entropy-based, and model-free policies.
5. Reproducible RunPod experiments with fixed configurations, seeds, logs, checkpoints, and
   statistical validation.

## Scientific Scope

The work studies:

- pulsed time-of-flight single-photon LiDAR;
- adaptive laser photon-budget allocation;
- Poisson photon-arrival statistics;
- atmospheric extinction and backscatter;
- detector efficiency, timing jitter, dead time, and dark counts;
- physics-guided artificial intelligence;
- low-reflectivity target detection;
- defense-oriented reconnaissance scenarios.

## Repository Structure

- `configs/`: version-controlled experiment configurations
- `data/`: raw, processed, synthetic, and external data
- `docs/`: hypotheses, decisions, protocols, and research notes
- `experiments/`: pilot, main, ablation, and robustness studies
- `figures/`: generated and publication-ready figures
- `literature/`: literature searches and structured paper notes
- `manuscript/`: LaTeX manuscript and submission materials
- `models/`: trained checkpoints and exported models
- `reports/`: experiment logs and validation reports
- `results/`: raw and processed numerical results
- `scripts/`: executable experiment entry points
- `src/`: reusable research software
- `tests/`: unit, integration, and reproducibility tests

## Reproducibility Principle

Every reported number must be recoverable from:

- a committed configuration;
- a specific Git commit;
- a recorded random seed;
- an environment description;
- a generated raw-result file;
- a deterministic analysis script.

No result should be entered manually into the manuscript.

## Intended Use

This repository is intended for controlled academic research and non-weaponized optical
sensing evaluation. It does not contain autonomous engagement, targeting, or weapon-control
functionality.
