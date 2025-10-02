# Four charts: DPW3 vs 2‑point PolyBLEP saw at A2 (110 Hz),
# for fs = 48 kHz and 96 kHz.
import numpy as np
import math
import matplotlib.pyplot as plt

def biquad_lpf(signal, fs, fc, Q=1/math.sqrt(2)):
    """RBJ cookbook low‑pass biquad (Direct Form I)."""
    w0 = 2.0 * math.pi * fc / fs
    cosw0 = math.cos(w0)
    sinw0 = math.sin(w0)
    alpha = sinw0 / (2.0 * Q)

    b0 = (1.0 - cosw0) * 0.5
    b1 = 1.0 - cosw0
    b2 = (1.0 - cosw0) * 0.5
    a0 = 1.0 + alpha
    a1 = -2.0 * cosw0
    a2 = 1.0 - alpha

    # normalize
    b0 /= a0; b1 /= a0; b2 /= a0
    a1 /= a0; a2 /= a0

    # Direct Form I processing
    y = np.zeros_like(signal, dtype=np.float64)
    x1 = x2 = y1 = y2 = 0.0
    for n, x0 in enumerate(signal):
        y0 = b0*x0 + b1*x1 + b2*x2 - a1*y1 - a2*y2
        y[n] = y0
        x2, x1 = x1, x0
        y2, y1 = y1, y0
    return y

def dpw_saw(f0, fs, order=2, dur_s=2.0):
    """DPW saw of given order (2 or 3 implemented)."""
    n = int(round(dur_s * fs))
    P0 = fs / f0
    dphi = f0 / fs
    # states for (order-1) iterated differences
    mem = [0.0] * (order - 1)
    x = np.zeros(n, dtype=np.float64)
    phi = 0.0
    for i in range(n):
        t = 2.0 * phi - 1.0  # phase mapped to [-1,1)
        if order == 2:
            p = t * t
        elif order == 3:
            p = t * t * t - t
        else:
            raise ValueError("order must be 2 or 3 for this demo")
        v = p
        for k in range(order - 1):
            d = v - mem[k]
            mem[k] = v
            v = d
        # normalization from JOS/Valimaki DPW: (P0/2)^(order-1)
        #y = v * ((P0 / 4.0) ** (order - 1))
        y = v * (np.pi**(order-1)) /(math.factorial(order) * (2*np.sin(np.pi/P0))**(order-1))
        x[i] = y
        phi += dphi
        if phi >= 1.0:
            phi -= 1.0
    x -= np.mean(x)
    return x

def polyblep(t, dt):
    # 2-point polyBLEP (simple form)
    if t < dt:
        x = t / dt
        return x + x - x * x - 1.0
    elif t > 1.0 - dt:
        x = (t - 1.0) / dt
        return x * x + 2.0 * x + 1.0
    else:
        return 0.0

def polyblamp(t, dt):
    # Integral-of-BLEP kernel, windowed to be 0 at both ends
    # piece 1: 0 <= t < dt
    if t < dt:
        # k(t) = ∫_0^t p(u) du - (t/dt)*∫_0^dt p(u) du
        # simplifies to:
        return (-t**3/(3*dt*dt) + t*t/dt - (2.0/3.0)*t)
    # piece 2: 1-dt < t <= 1
    elif t > 1.0 - dt:
        # k(t) = ∫_{1-dt}^t p(u) du - ((t-(1-dt))/dt) * ∫_{1-dt}^1 p(u) du
        # simplifies to:
        return ((t-1.0)**3/(3*dt*dt) + (t-1.0)**2/dt + (t-1.0) - t/3.0 + 1.0/3.0)
    else:
        return 0.0
    
def polyblep_saw(f0, fs, dur_s=2.0):
    n = int(round(dur_s * fs))
    dt = f0 / fs
    x = np.zeros(n, dtype=np.float64)
    t = 0.0
    for i in range(n):
        y = 2.0 * t - 1.0      # naive saw
        y -= polyblep(t, dt)   # remove step
        #y += polyblamp(t,dt)
        x[i] = y
        t += dt
        if t >= 1.0:
            t -= 1.0
    x -= np.mean(x)
    return x

def single_sided_spectrum_dbfs(sig, fs):
    n = len(sig)
    w = np.hanning(n)
    cg = np.sum(w) / n
    X = np.fft.rfft(sig * w)
    amp = (2.0 / (n * cg)) * np.abs(X)
    eps = 1e-20
    db = 20.0 * np.log10(np.maximum(amp, eps))
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    return freqs, db

def plot_spec(sig, fs, title):
    freqs, db = single_sided_spectrum_dbfs(sig, fs)
    plt.figure()
    plt.plot(freqs, db)
    plt.axvline(20000.0, linestyle='--')  # in-band marker
    plt.axvline(28000.0, linestyle='--')  # in-band marker
    plt.xlim(0, 48000.0)
    plt.ylim(-160, 0)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dBFS)")
    plt.title(title)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)

f0 = 110.0*2**2
fc = 24000
for fs in (2*48000, 4*96000):
    #sig_dpw3 = dpw_saw(f0, fs, order=3, dur_s=2.0)
    #plot_spec(sig_dpw3, fs, f"DPW2 Saw — A2 (110 Hz) @ fs={fs/1000:.0f} kHz")
    sig_pblep = polyblep_saw(f0, fs, 2.0)
    #sig_pblep_f = biquad_lpf(sig_pblep, fs, fc)
    #for i in range(1,9):
    #    sig_pblep_f = biquad_lpf(sig_pblep_f, fs, fc)
    plot_spec(sig_pblep, fs, f"2‑point PolyBLEP Saw — A2 ({f0:.0f} Hz) @ fs={fs/1000:.0f} kHz")

plt.show()
