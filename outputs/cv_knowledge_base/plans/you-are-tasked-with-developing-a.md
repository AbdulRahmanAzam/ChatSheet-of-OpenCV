# Plan: Wavelet denoising and anomaly detection on a signal

> You are tasked with developing a system to monitor sensor data from industrial machines and identify anomalies in real-time using the wavelet transformation.

## What the task needs

- **Goal:** denoise the signal and flag abnormal points
- **Input:** a 1-D sensor signal
- **Method:** Wavelet denoising and anomaly detection on a signal (chosen by cue words: 'wavelet', 'sensor', 'anomal')
- **Also considered:** Object recognition with SIFT, ratio test and RANSAC homography (2), Geometric transformation (1)
- **Limits to state in your answer:** Thresholding removes some detail and may suppress anomalies themselves. Boundary padding and scale affect output. A single threshold cannot detect every anomaly type. Slow drift may stay in the approximation.

## Steps at a glance

1. Load the sensor readings
2. Haar wavelet denoise and residual-based anomaly detection
3. Plot the signal, the denoised curve and the anomalies

## Step 1: Load the sensor readings

One reading per line (CSV). The wavelet steps need a 1-D float array.

```python
signal = np.loadtxt('sensor.csv', delimiter=',')
```

**Watch out:** Thresholding removes some detail and may suppress anomalies themselves. Boundary padding and scale affect output.

*Source: Lab Manual 06 p.3; Lab Manual 06 p.4; Lab 06 Tasks p.1*

## Step 2: Haar wavelet denoise and residual-based anomaly detection

The wavelet transform splits the signal into a smooth part and detail parts. Small details are noise and are shrunk to zero; the smooth reconstruction is the normal behaviour. Points far from it (more than Z_LIMIT robust standard deviations) are anomalies. MAD is used instead of std so the spikes do not hide themselves.

Parameters:
- `LEVELS = 3`: wavelet decomposition levels
- `Z_LIMIT = 4.0`: robust z-score limit for an anomaly

```python
x = np.asarray(signal, float)
approx, details = x[: len(x) // 2 ** LEVELS * 2 ** LEVELS], []
for _ in range(LEVELS):                                   # forward Haar transform
    details.append((approx[0::2] - approx[1::2]) / np.sqrt(2))
    approx = (approx[0::2] + approx[1::2]) / np.sqrt(2)
sigma = np.median(np.abs(details[0])) / 0.6745             # noise level from the finest details
thr = sigma * np.sqrt(2 * np.log(len(x)))                  # universal threshold
for d in reversed(details):                                # soft-threshold details, inverse transform
    d = np.sign(d) * np.maximum(np.abs(d) - thr, 0)
    up = np.empty(2 * len(approx)); up[0::2] = (approx + d) / np.sqrt(2); up[1::2] = (approx - d) / np.sqrt(2)
    approx = up
denoised = approx
residual = x[: len(denoised)] - denoised
mad = np.median(np.abs(residual - np.median(residual))) / 0.6745
anomalies = np.flatnonzero(np.abs(residual - np.median(residual)) > Z_LIMIT * max(mad, 1e-9))
report['anomalies'] = anomalies.tolist()
```

**Watch out:** The signal length must be divisible by 2**LEVELS for this Haar code; extra samples at the end are dropped.

**From the course:** Wavelet transformation analyzes a signal or image in terms of its frequency components at different scales. *(Lab Manual 06 p.3)*

**From the course:** The inverse wavelet transform reconstructs a signal or image from its wavelet coefficients; it is crucial for compression and denoising. *(Lab Manual 06 p.4)*

*Source: Lab Manual 06 p.3; Lab Manual 06 p.4; Lab 06 Tasks p.1; Lab 06 Tasks p.2*

## Step 3: Plot the signal, the denoised curve and the anomalies

Matplotlib shows the original readings, the wavelet-denoised trend and red dots at anomalies.

```python
plt.plot(signal, lw=0.8, label='sensor')
plt.plot(denoised, label='denoised')
plt.scatter(anomalies, signal[anomalies], c='r', label='anomaly')
plt.legend(); plt.show()
```

**Watch out:** Matplotlib expects RGB; convert BGR first. Without vmin/vmax, grayscale panels auto-stretch and are not comparable.

*Source: KB supplement (matplotlib)*

## Closest worked lab task: Wavelet sensor anomalies (L06-T03)

The knowledge base has a tested solution for a similar lab task (match 0.2). Its steps:

1. Decompose clean baseline with Haar DWT.
2. Estimate noise from detail coefficients and soft-threshold for a denoised output.
3. Separately reconstruct approximation-only signals and compute detail residuals.
4. Calibrate a cutoff using clean-baseline residual MAD.
5. Apply the same decomposition to the test signal and return deviations.

Limits: Signal-processing appendix preserved from course. Baseline assumed representative; broad/slow anomalies can be missed. Soft-threshold denoising residuals are not used as anomaly scores because they can cap retained impulses.

Code (`solutions/lab06.py`, `lab06.task03_wavelet_anomalies`; helpers come from `solutions/cv_core.py`):

```python
def task03_wavelet_anomalies(signal,clean_baseline,levels=3,mad_factor=5):
    baseline=np.asarray(clean_baseline,float); signal=np.asarray(signal,float)
    _,threshold=haar_denoise(baseline,levels)
    # Score the detail component against an approximation-only reconstruction.
    # A soft-threshold-denoising residual is capped by its threshold and can hide
    # a large retained impulse, so it must not be the anomaly statistic here.
    baseline_coarse,_=haar_denoise(baseline,levels,float('inf'))
    residual=baseline-baseline_coarse; center=float(np.median(residual))
    sigma=float(np.median(np.abs(residual-center))/0.67448975)
    cutoff=max(mad_factor*sigma,1e-8)
    denoised,_=haar_denoise(signal,levels,threshold)
    coarse,_=haar_denoise(signal,levels,float('inf'))
    detail=signal-coarse
    score=np.abs(detail-center)
    return {'denoised':denoised,'residual':detail,'score':score,
            'anomaly_indices':np.flatnonzero(score>cutoff),'cutoff':cutoff,'wavelet_threshold':threshold}
```

## Full script

Every step above, in order, as one runnable file (tune the PARAMETERS block for your images):

```python
"""Wavelet denoising and anomaly detection on a signal.

Task: You are tasked with developing a system to monitor sensor data from industrial machines and identify
      anomalies in real-time using the wavelet transformation.

Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().
"""
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMETERS: tune these for your images ----
LEVELS = 3                 # wavelet decomposition levels
Z_LIMIT = 4.0              # robust z-score limit for an anomaly


def run(signal):
    """Run the plan on one input; returns (stages to display, report of numbers)."""
    stages, report = {}, {}
    # Step 2: Haar wavelet denoise and residual-based anomaly detection
    x = np.asarray(signal, float)
    approx, details = x[: len(x) // 2 ** LEVELS * 2 ** LEVELS], []
    for _ in range(LEVELS):                                   # forward Haar transform
        details.append((approx[0::2] - approx[1::2]) / np.sqrt(2))
        approx = (approx[0::2] + approx[1::2]) / np.sqrt(2)
    sigma = np.median(np.abs(details[0])) / 0.6745             # noise level from the finest details
    thr = sigma * np.sqrt(2 * np.log(len(x)))                  # universal threshold
    for d in reversed(details):                                # soft-threshold details, inverse transform
        d = np.sign(d) * np.maximum(np.abs(d) - thr, 0)
        up = np.empty(2 * len(approx)); up[0::2] = (approx + d) / np.sqrt(2); up[1::2] = (approx - d) / np.sqrt(2)
        approx = up
    denoised = approx
    residual = x[: len(denoised)] - denoised
    mad = np.median(np.abs(residual - np.median(residual))) / 0.6745
    anomalies = np.flatnonzero(np.abs(residual - np.median(residual)) > Z_LIMIT * max(mad, 1e-9))
    report['anomalies'] = anomalies.tolist()
    return stages, report, denoised, anomalies


def show(original, stages):
    """Original plus every stage in one matplotlib figure (BGR converted to RGB)."""
    items = [('Original', original)] + [(k, v) for k, v in stages.items() if isinstance(v, np.ndarray) and v.ndim in (2, 3)]
    cols = min(3, len(items)); rows = (len(items) + cols - 1) // cols
    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, (name, image) in enumerate(items, 1):
        plt.subplot(rows, cols, i)
        if image.ndim == 3:
            plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        else:
            plt.imshow(image, cmap='gray')
        plt.title(name); plt.axis('off')
    plt.tight_layout(); plt.show()

def main():
    signal = np.loadtxt(sys.argv[1] if len(sys.argv) > 1 else 'sensor.csv', delimiter=',').ravel()
    stages, report, denoised, anomalies = run(signal)
    print('anomaly indices:', report['anomalies'])
    plt.figure(figsize=(12, 4))
    plt.plot(signal, lw=0.8, label='sensor')
    plt.plot(denoised, label='denoised')
    plt.scatter(anomalies, signal[anomalies], c='r', zorder=3, label='anomaly')
    plt.legend(); plt.title('Sensor data, denoised signal and anomalies'); plt.show()

if __name__ == '__main__':
    main()
```
