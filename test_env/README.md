---
title: ICU Bed Allocation Environment
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# ICU Bed Allocation Environment

This environment defines a reinforcement learning task for ICU triage and bed allocation.

## Components

- ICUAction / ICUObservation / ICUState / Patient in models.py
- ICUEnv client implementation in client.py
- Environment server implementation under server/

## Usage

The environment is configured as an OpenEnv environment and can be run via the OpenEnv tooling or by using the ICUEnv client directly from Python.
