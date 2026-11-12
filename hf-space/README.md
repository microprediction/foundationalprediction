---
title: Foundational Prediction
emoji: 📈
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: true
license: mit
short_description: Which model for the next data point? One-step leaderboards
tags:
  - leaderboard
  - time-series
  - time-series-forecasting
  - probabilistic-forecasting
---

# Foundational Prediction

Leaderboards for the question a deployed forecaster actually faces: which
model best predicts the next observation of a stream?

- **One horizon at a time.** The next observation is scored alone, never
  pooled with later steps.
- **Full predictive densities, scored by log likelihood**, with CRPS and
  interval coverage alongside. Quantile models are turned into densities by
  a stated reconstruction.
- **Paired, per series.** Every model is compared with a reference on
  identical series and windows; a per-series Diebold–Mariano test decides
  each series, and the board reports wins, draws and losses.
- **Training-data overlap disclosed per model.**

The roster covers Chronos-2, TimesFM 3.0, Timestral, t0, TimeH2O, Toto 2.0,
TiRex-2, Moirai 2.0, TTM-R3, Simons, ci-proto, TinyCast and classical
baselines, against `laplace` and `granger` from
[skaters](https://github.com/microprediction/skaters), on FRED economic and
price series and the M4 hourly set.

This Space mirrors [foundationalprediction.com](https://foundationalprediction.com)
and is rebuilt from the same source on every update
([source](https://github.com/microprediction/foundationalprediction)).
