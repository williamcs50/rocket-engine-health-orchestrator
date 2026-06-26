# Rocket Engine Health Orchestrator: Design Rationale

**Author:** William Opyrchal  
**Date:** June 26, 2026

## Summary

This document lays out the architecture I came up with for a Rocket Engine Health Orchestrator. The system uses per-channel encoders to produce fixed-length embeddings for individual sensor signals, self-attention in a fusion layer to discover cross-channel patterns, and cross-attention to a command history sequence to tell the difference between normal responses to commands and actual faults. The output is a health score with uncertainty, a short ranked list of contributing channels, and attention weights so you can dig into what the model flagged. I also want a visualization layer that maps the output back to the physical engine, because the system is only useful if an engineer can quickly make sense of what it is saying.

The main priority was keeping the architecture modular and easy to iterate on. I am starting with uniform encoders across all channels and only adding complexity when data shows it is necessary.

One open question I want to test early is whether cross-attention over a command sequence is good enough to capture command-to-response dynamics, or whether a small recurrent model would do better. That is the first experiment I want to run once the pipeline is working end to end.

## How I Got Here

I started with the simplest possible case: one sensor channel. What does the model actually need to see? I decided it needed a short window of recent timesteps so it could learn normal patterns, trends, and rates of change on its own. That became the foundation for the per-channel encoders.

When scaling to many channels, I considered grouping similar channels together and giving each group its own encoder. I rejected that because per-channel encoders give the fusion layer something cleaner to work with: each embedding maps to one physical quantity, so the attention weights produced by the fusion layer map directly back to individual sensors. That is what makes the output legible to an operator. A group embedding is a blend, and when the fusion layer attends to it, the engineer cannot ask what happened in any specific channel within that group. Per-channel keeps the interpretability story intact end to end. I also considered a single monolithic transformer over all raw channels concatenated together, but rejected it because the model would have to deal with mismatched scales, units, and physics all at once.

The biggest practical challenge was avoiding false alarms during commanded changes. When the controller issues a throttle-down or valve movement, a lot of sensor channels will legitimately shift at the same time. Without context, those normal movements look like faults. I considered concatenating command signals as extra input channels, but rejected that because it gives commands no structural priority and the model could learn to ignore them. I also considered baking command context directly into each encoder, but rejected that too because it would couple the encoders more tightly to command logic and make the architecture less modular. Instead I kept the encoders focused purely on sensor data and handled command context in the fusion layer through cross-attention to a short command history sequence.

## Output and Visualization

I want the output to actually be useful, not just technically correct. A health score by itself is not enough. The system should also return a short ranked list of the channels that contributed most to the score, along with the attention weights, so an engineer can quickly see what the model flagged as unusual. The visualization layer matters a lot here because staring at raw weights is not practical under time pressure. Mapping everything back to the physical engine layout makes the output much faster to interpret.

One thing I want to be careful about is calibration. The health score should be well-calibrated so a reported confidence level actually matches reality. The way I am thinking about it: a score of 0.85 should mean the current engine state is more unusual than 85% of normal examples the model has seen under similar commanded conditions. Most neural networks are overconfident by default, so I plan to apply post-hoc calibration (likely temperature scaling) after training. This matters because an overconfident false negative is especially dangerous here. I would rather the model surface uncertainty than hide it.

## The Four Floor Commitments

These are the constraints I want to hold regardless of how the design evolves:

- The orchestrator needs to actually catch coupled, sub-threshold faults that conventional per-channel redlines would miss on the same data. That is the baseline comparison I need to demonstrate.
- Training, calibration, and fault-injection data have to be generated separately and kept disjoint. The model should only see commands and telemetry at inference time.
- Attention weights indicate which channels were relevant, not what caused the fault. I want to be honest about that distinction in how the output is presented.
- The system advises, it does not act. Deterministic redline protection keeps all authority.

## Comparison with the Napkin

After I had worked out the core pipeline on my own, I compared it against the napkin design and made a few observations.

We aligned on the core structure: per-channel encoders feeding into an attention-based fusion layer, command conditioning to separate faults from commanded responses, a calibrated advisory output, and the idea that the output needs to be interpretable enough for an engineer to act on quickly.

On visualization, I had already concluded it was a requirement: without a spatial layout that maps the verdict back to the physical engine, the output stays abstract and gets ignored under pressure. What the napkin provided was a specific layout for solving that problem: the ring-style design. I recognized it as the right answer to a problem I had already identified. On the longer-term roadmap, the napkin pointed toward graph attention over a learned channel-influence topology, which felt like a natural next rung once the flat attention fusion is working.

Where I diverged: I chose to start with uniform encoders rather than building in encoder diversity from the start, and I chose cross-attention to a command sequence in the fusion layer instead of using recurrence for command-to-response modeling. Both decisions came from wanting to keep the early system simpler and easier to debug and calibrate.

## Key Decisions and Open Questions

The two decisions I feel most confident about are starting with uniform encoders and handling command context in the fusion layer rather than inside each encoder. Both keep the system simpler and more modular early on, and I can add complexity later if I need to.

The main open question is whether cross-attention over a command sequence is actually good enough to learn command-to-response dynamics, or whether a small recurrent model would do better. That is the first experiment I want to run once the full pipeline is working end to end.

## Why This Document

I wanted to record not just what I decided but what I considered and rejected and why. The outcome is in the code. The reasoning needs to be somewhere else.