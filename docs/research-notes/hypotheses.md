# Research Hypotheses

## H1: Photon Efficiency

The proposed physics-guided adaptive allocation policy will achieve a specified probability
of detection using fewer transmitted photons than uniform allocation.

## H2: Detection Performance

At an equal photon budget, the proposed policy will improve probability of detection while
maintaining the same false-alarm probability.

## H3: Ranging Performance

At an equal photon budget, the proposed policy will reduce target-range RMSE relative to
fixed and heuristic allocation strategies.

## H4: Robustness

The proposed policy will preserve its advantage under previously unseen atmospheric
attenuation, background illumination, reflectivity, timing jitter, and detector-efficiency
conditions.

## H5: Calibration

The model's posterior uncertainty will be positively associated with its observed ranging
and detection errors.

## Primary Endpoint

Percentage reduction in the transmitted photon budget needed to achieve:

- probability of detection >= 0.90;
- false-alarm probability <= 0.05;
- range RMSE below a predefined engineering threshold.

## Secondary Endpoints

- ROC-AUC;
- average precision;
- range MAE and RMSE;
- time-to-detection;
- photon efficiency;
- uncertainty-calibration error;
- inference time;
- GPU memory usage.
