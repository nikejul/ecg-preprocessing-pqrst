# ECG Signal Pre-processing and PQRST Detection

Final project, D3 Telecommunication Engineering, Politeknik Negeri Bandung (2025).

A Python system that cleans noisy ECG signals and detects the P, Q, R, S, and T waves. It works on recorded data (Excel files) and on live signals from an AD8232 module connected to an Arduino.

> This is an academic prototype, not a medical device. It must not be used for diagnosis.

## Pipeline

```
Raw ECG -> Band-pass filter -> Baseline correction -> Wavelet denoising -> Smoothing -> PQRST detection
```

| Stage | Method |
|---|---|
| Band-pass filter | Butterworth, 6th order, 0.5-40 Hz |
| Baseline correction | Gaussian filter |
| Denoising | Wavelet with adaptive (soft) threshold |
| Smoothing | Bartlett window |
| Detection | Modified Pan-Tompkins (P, Q, R, S, T) |
| Output | Heart rate (BPM), R-R interval, noise-reduction percentage |

## Features

- **File mode:** drop an Excel file with ECG samples and view the signal at every processing stage
- **Real-time mode:** stream ECG from an AD8232 through an Arduino over USB serial (COM port selectable from a dropdown)
- Automatic BPM and noise-reduction percentage
- Sampling rate: 100 Hz

## Hardware

- AD8232 ECG module with electrodes
- Arduino (USB serial, 9600 baud)

## Requirements

- Python 3.9+
- numpy, pandas, scipy, PyWavelets, matplotlib, pyserial, openpyxl

```bash
pip install numpy pandas scipy PyWavelets matplotlib pyserial openpyxl
```

Tkinter is included with most Python installations.

## Usage

1. Clone the repository:
```bash
   git clone https://github.com/nikejul/ecg-preprocessing-pqrst.git
   cd ecg-preprocessing-pqrst
```
2. **File mode:** run the drop-file application and drag in an Excel file containing the ECG samples.
3. **Real-time mode:** upload the Arduino sketch, connect the AD8232, run the real-time application, choose the COM port, and start recording.

Make sure the baud rate in the Python code matches the one in the Arduino sketch (9600).

## Results

| Condition | PQRST detection success |
|---|---|
| High-quality, low-noise signal | 94% |
| Normal condition | 85% |
| Signal with motion artifacts | 65% |

Tests covered several AD8232 modules, healthy and unhealthy subjects, and resting versus moving conditions.

## Limitations

- T-wave detection is the weakest part (about 80% missed), because its shape varies and its amplitude is often low.
- Performance drops sharply with motion artifacts.
- Loose jumper wires on the AD8232 and Arduino interrupted the signal during movement tests.
- The application occasionally froze during long sessions.

## Future work

- Adaptive thresholding or template matching for more robust detection
- Testing on a wider, more diverse ECG dataset
- Extra features such as QT interval and arrhythmia detection

## Author

Nike Julian Nensi, D3 Telecommunication Engineering, Politeknik Negeri Bandung.
Supervisors: Prof. Ir. Hertog Nugroho, M.Sc., Ph.D. and Dr. Eril Mozef, MS., DEA.
