import numpy as np
import matplotlib.pyplot as plt

# Set the variables
sin_amplitude = 1.25    # Amplitude for sine component
cos_amplitude = 0.75    # Amplitude for cosine component
freq_sin = 30           # Frequency for sine component
freq_cos = 130          # Frequency for cosine component
noise_level = 2.5       # Uniform noise level

# Create a simple signal with two frequencies
dt = 0.001               # Sampling interval
t = np.arange(0, 1, dt)  # Time vector
n = len(t)               # Number of samples
f_clean = sin_amplitude * np.sin(2 * np.pi * freq_sin * t) + cos_amplitude * np.cos(2 * np.pi * freq_cos * t)  # Original signal
f = f_clean + noise_level * np.random.uniform(-1, 1, n)  # Add noise

# Compute FFT
fhat = np.fft.fft(f, n)                     # FFT of the noisy signal
PSD = np.real(fhat * np.conj(fhat) / n)     # Power spectrum (normalized)
freq = (1 / (dt * n)) * np.arange(n)        # Frequency vector

# Keep only the first half of positive frequencies (Nyquist-Shannon Sampling Theorem)
half_n = n // 2                         
PSD = PSD[:half_n]
freq = freq[:half_n]
fhat[1:half_n] = fhat[1:half_n] * 2         # Scaling by two except the DC

# Filter frequencies with power below a certain threshold
threshold = 0.25 * max(PSD)                 # Threshold for significant frequencies
indices = PSD > threshold
PSD_clean = indices * PSD                   # Filtered power spectrum
fhat_clean = indices * fhat[:half_n]        # Filtered FFT

# Inverse FFT for the filtered signal (using only positive frequencies)
ffilt = np.fft.ifft(fhat_clean, n).real     # Reconstruct signal (real part only)

# Returning the original clean function
dominant_indices = np.where(PSD_clean > 0)[0]                       # Find indices where PSD_clean is non-zero
dominant_frequencies = freq[dominant_indices]                       # Get the corresponding frequencies
dominant_amplitudes = np.abs(fhat_clean[dominant_indices]) / n      # Extract the dominant amplitudes
dominant_phases = np.angle(fhat_clean[dominant_indices])            # Extract the dominant phases
cosine_components = dominant_amplitudes * np.cos(dominant_phases)   # Calculate the cosine/sine components
sine_components = -dominant_amplitudes * np.sin(dominant_phases)

# Print the dominant frequencies, their amplitudes, phases, and sine/cosine components
for i in range(len(dominant_frequencies)):
    print(f"Frequency: {dominant_frequencies[i]} Hz, Amplitude: {dominant_amplitudes[i]}, Phase: {dominant_phases[i]} radians, Cosine component: {cosine_components[i]}, Sine component: {sine_components[i]}")

# Plot three results
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t, f, label='Noisy Signal', color='r', linewidth=1.5)
plt.plot(t, f_clean, label='Original Signal', color='k', linewidth=1)
plt.title('Time Domain Signals')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(freq, PSD, label='Noisy Power Spectrum', color='r')
plt.plot(freq, PSD_clean, label='Filtered Power Spectrum', color='b')
plt.title('Power Spectrum')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, f_clean, label='Original Signal', color='k', linewidth=1)
plt.plot(t, ffilt, label='Filtered Signal', color='b', linewidth=1.5)
plt.title('Filtered Time Domain Signal')
plt.legend()

plt.tight_layout()
plt.show()