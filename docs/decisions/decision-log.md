
## Decision 001 — Physics model before AI development

**Status:** Accepted

**Decision:** Implement and validate the photon-return forward model before
developing adaptive policies or reinforcement-learning components.

**Rationale:** The proposed contribution depends on physically meaningful
photon allocation. AI results cannot be trusted unless atmospheric loss,
geometric spreading, detector effects, background photons and time-of-flight
histograms are validated first.

**Current model includes:**

- inverse-square geometric spreading;
- two-way Beer-Lambert attenuation;
- target reflectivity;
- receiver aperture;
- optical and detector efficiencies;
- Gaussian timing response;
- background and dark-count photons;
- Poisson photon sampling.

**Deferred extensions:**

- beam divergence;
- target bidirectional reflectance;
- incidence angle;
- speckle;
- spatially varying backscatter;
- detector dead-time saturation;
- afterpulsing;
- moving-target effects.
