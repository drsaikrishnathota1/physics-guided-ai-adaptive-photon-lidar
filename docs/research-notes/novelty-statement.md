# Preliminary Novelty Statement

## Proposed Innovation

A physics-guided AI policy dynamically allocates a finite photon budget across spatial
regions and sequential LiDAR measurements according to posterior target probability,
range uncertainty, atmospheric transmission, and detector limitations.

## Difference from Conventional Approaches

Conventional single-photon LiDAR methods commonly:

1. distribute photons uniformly;
2. apply fixed raster-scanning patterns;
3. reconstruct depth after all measurements are collected;
4. use AI only for post-processing.

The proposed method uses AI within the optical sensing loop. It chooses the next photon
allocation before the next measurement is acquired.

## Core Testable Claim

At matched probability of detection and false-alarm probability, the proposed adaptive
policy requires fewer transmitted photons than uniform, random, greedy, and entropy-only
allocation strategies.

## Secondary Claims

- Improved ranging accuracy under a fixed photon budget.
- Improved robustness under atmospheric and detector mismatch.
- Better calibration between predicted uncertainty and actual ranging error.
- Reduced time-to-detection for spatially sparse targets.

## Claims Explicitly Excluded

- No claim of quantum advantage.
- No claim that single-photon detection alone constitutes quantum LiDAR.
- No claim of field deployment readiness.
- No claim of covert operation unless interception probability is explicitly modeled.
