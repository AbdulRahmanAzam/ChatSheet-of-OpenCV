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
