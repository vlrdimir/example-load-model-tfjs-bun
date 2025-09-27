# research-model-garbage

This repository contains TypeScript script examples that demonstrate how to load and run a pre-trained image classification model using TensorFlow.js in the Bun runtime environment.

## Features

- Loading TensorFlow.js model (`GraphModel`) from URL.
- Explicit CPU backend setup using `@tensorflow/tfjs-backend-cpu`.
- Reading and decoding local JPEG images into tensors.
- Image preprocessing (resizing) to match model input.
- Making predictions and displaying class with highest probability.
