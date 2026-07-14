# Core Reference Map

## Scope

This reference set supports a short communication on physics-guided adaptive
photon allocation for single-photon LiDAR. References should be used only
for claims they directly support.

---

## 1. Foundational Photon-Efficient Imaging

### Kirmani et al. — First-Photon Imaging

Ahmed Kirmani, Dheera Venkatraman, Dongeek Shin, Andrea Colaco,
Franco N. C. Wong, and Vivek K. Goyal.

"First-Photon Imaging."

Science, volume 343, issue 6166, pages 58–61, 2014.

DOI: 10.1126/science.1246775

Use for:

- imaging with extremely sparse detected photons;
- photon-efficient depth and reflectivity estimation;
- motivation for operating in photon-starved conditions.

Suggested manuscript location:

- Introduction, paragraph 1.

---

### Shin et al. — Photon-Efficient Imaging with a Single-Photon Camera

Dongeek Shin, Feihu Xu, Dheera Venkatraman, Rudi Lussana,
Federica Villa, Franco Zappa, Vivek K. Goyal,
Franco N. C. Wong, and Jeffrey H. Shapiro.

"Photon-Efficient Imaging with a Single-Photon Camera."

Nature Communications, volume 7, article 12046, 2016.

DOI: 10.1038/ncomms12046

Use for:

- photon-efficient three-dimensional imaging;
- single-photon camera operation;
- computational recovery under low photon counts.

Suggested manuscript location:

- Introduction;
- Related-work discussion.

---

## 2. Long-Range Single-Photon LiDAR

### McCarthy et al. — Kilometre-Range Depth Imaging

Aongus McCarthy, Nathan J. Krichel, Nathan R. Gemmell,
Ximing Ren, Michael G. Tanner, Sander N. Dorenbos,
Val Zwiller, Robert H. Hadfield, and Gerald S. Buller.

"Kilometre-Range, High Resolution Depth Imaging via
1560 nm Wavelength Single-Photon Detection."

Optics Express, volume 21, issue 7, pages 8904–8915, 2013.

DOI: 10.1364/OE.21.008904

Use for:

- long-range photon-counting depth imaging;
- operation near the 1550 nm telecommunications band;
- defense and reconnaissance motivation.

Suggested manuscript location:

- Introduction, paragraph 1;
- Discussion.

---

### Hadfield et al. — Long-Range Imaging and Sensing Review

Robert H. Hadfield, Jonathan Leach, Fiona Fleming,
Douglas J. Paul, and Chee Hing Tan.

"Single-Photon Detection for Long-Range Imaging and Sensing."

Optica, 2023.

Use for:

- modern detector technologies;
- long-range single-photon sensing;
- SPAD and superconducting detector tradeoffs;
- detector efficiency, timing resolution, and dark counts.

The final volume, issue, pages, and DOI must be checked before submission.

Suggested manuscript location:

- Introduction;
- Detector-model subsection;
- Discussion.

---

## 3. Bayesian Sparse-Photon Processing

### Altmann et al. — Sparse Waveform Analysis

Yoann Altmann, Ximing Ren, Aongus McCarthy,
Gerald S. Buller, and Steve McLaughlin.

"Lidar Waveform-Based Analysis of Depth Images Constructed
Using Sparse Single-Photon Data."

IEEE Transactions on Image Processing, 2016.

Use for:

- Poisson photon-count likelihoods;
- Bayesian depth and reflectivity estimation;
- unknown background treatment;
- sparse time-correlated photon-count data.

Final bibliographic details and DOI must be verified.

Suggested manuscript location:

- Sequential target-probability update;
- Introduction.

---

### Altmann et al. — Bayesian Target Detection

Yoann Altmann, Ximing Ren, Aongus McCarthy,
Gerald S. Buller, and Steve McLaughlin.

"Robust Bayesian Target Detection Algorithm for Depth Imaging
from Sparse Single-Photon Data."

Use for:

- target-present and target-absent hypotheses;
- Bayesian model selection;
- background-aware target detection;
- posterior target probability.

Final journal, volume, pages, year, and DOI must be verified.

Suggested manuscript location:

- Sequential target-probability update.

---

## 4. Real-Time Single-Photon Reconstruction

### Tachella et al. — Real-Time Reconstruction

Julian Tachella, Yoann Altmann, Nicolas Mellado,
Aongus McCarthy, Rachael Tobin, Gerald S. Buller,
Jean-Yves Tourneret, and Stephen McLaughlin.

"Real-Time 3D Reconstruction from Single-Photon Lidar Data
Using Plug-and-Play Point Cloud Denoisers."

Nature Communications, volume 10, article 4984, 2019.

DOI: 10.1038/s41467-019-12943-0

Use for:

- practical outdoor single-photon LiDAR;
- reconstruction in daylight;
- computational efficiency;
- processing of complex and cluttered scenes.

Suggested manuscript location:

- Introduction;
- Discussion and future work.

---

### Altmann et al. — Online Dynamic Reconstruction

Yoann Altmann, Stephen McLaughlin, and Michael E. Davies.

"Fast Online 3D Reconstruction of Dynamic Scenes from
Individual Single-Photon Detection Events."

Use for:

- sequential processing;
- individual photon-event processing;
- dynamic-scene reconstruction;
- avoiding long histogram integration times.

Final publication details and DOI must be verified.

Suggested manuscript location:

- Discussion and future work.

---

## 5. SPAD Detector Modeling

### Bruschini et al. — SPAD Review

Claudio Bruschini, Harald Homulle, Ivan Michel Antolovic,
Samuel Burri, and Edoardo Charbon.

"Single-Photon Avalanche Diode Imagers in Biophotonics:
Review and Outlook."

Light: Science & Applications, volume 8, article 87, 2019.

DOI: 10.1038/s41377-019-0191-5

Use for:

- SPAD operating principles;
- dark-count noise;
- dead time;
- timing jitter;
- afterpulsing;
- detector-array considerations.

Although focused partly on biophotonics, the detector physics is relevant.

Suggested manuscript location:

- Detector-model subsection.

---

## 6. Information-Guided Adaptive Sensing

The proposed allocation score is a lightweight heuristic combining:

- posterior entropy;
- predicted signal-to-background ratio;
- measurement-count exploration.

The manuscript must not describe it as exact expected mutual information
unless exact posterior predictive integration is performed.

Use terminology such as:

- physics-guided information allocation;
- information-inspired adaptive sensing;
- posterior-uncertainty allocation;
- physics-aware sequential measurement selection.

A general information-theory or Bayesian experimental-design reference may
be included to explain entropy and information-guided measurement selection,
but it should not be presented as prior single-photon LiDAR implementation
unless the cited work actually applies the method to LiDAR.

---

## Recommended Reference Count

For the short communication:

- 15–20 total references;
- approximately 8–10 single-photon LiDAR papers;
- 2–3 SPAD or detector papers;
- 2–3 adaptive sensing or Bayesian design papers;
- 2–3 atmospheric or laser-radar modeling sources.

Avoid adding unrelated AI, quantum computing, autonomous vehicle, or defense
papers merely to increase the reference count.
