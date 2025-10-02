# FFT of the 32-segment quarter-wave cosine approximation ("endpoint+midpoint" version)
# plus: per-section quadratic least-squares ("polyfit") option.
# f0 = 110 Hz, Fs = 48 kHz, using coherent length (11 periods = 4800 samples).

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from fractions import Fraction

# ------------------ Controls ------------------
METHOD = "constrained polyfit"   # "interp" for endpoint+midpoint; "polyfit" for LS quadratic
SEGMENTS = 32
NSAMP_LS = 200       # samples per segment for the LS fit (dense grid)
FRACTION = False
SHOW_FRACTION = False
DEN_WIDTH = 5
QUANTIZE = True
MULTIPLIER_WIDTH = 18
ADDER_WIDTH = 24
# ----------------------------------------------

Fs = 48000.0*2
f0 = 110.0*2**0
Nper = 11000                  # coherent: 11 periods0
N = int(Fs/f0 * Nper)         # 4800 samples
t = np.arange(N)/Fs
phase = (f0*t) % 1.0          # x in [0,1)

# Build the quarter-wave approximation
S = SEGMENTS
h = 0.25 / S
nodes = np.linspace(0.0, 0.25, S+1)

def cos_q(x):
    return np.cos(2*np.pi*x)

# ---------- Method 1: endpoint+midpoint interpolating quadratic ----------
# p(u) = A + B*u + C*u^2, u in [0,1]
coeffs_interp = []
for j in range(S):
    x0 = nodes[j]; x1 = nodes[j+1]; xm = 0.5*(x0+x1)
    y0 = cos_q(x0); y1 = cos_q(x1); ym = cos_q(xm)
    A = y0
    B = (y1 - y0) + (ym - 0.5*(y0+y1))
    C = -(ym - 0.5*(y0+y1))
    # print(f"{x0*128}/128: {C}x**2+{B}x+{A}")
    coeffs_interp.append((x0,A,B,C))
coeffs_interp = np.array(coeffs_interp, dtype=float)

def eval_quarter_interp(x):
    # x in [0,0.25)
    j = np.minimum((x / h).astype(int), S-1)
    x0 = coeffs_interp[j,0]; A = coeffs_interp[j,1]; B = coeffs_interp[j,2]; C = coeffs_interp[j,3]
    u = (x - x0) / h
    return A + B*u + C*(u*u)

# ---------- Method 2: per-section quadratic least-squares (polyfit) ----------
# Fit p(x) = c0 + c1*x + c2*x^2 on each segment using dense sampling
coeffs_ls = []
for j in range(S):
    x0 = nodes[j]; x1 = nodes[j+1]
    xs = np.linspace(x0, x1, NSAMP_LS, endpoint=True)
    ys = cos_q(xs)
    c2, c1, c0 = np.polyfit(xs, ys, deg=2)   # np.polyfit returns [c2,c1,c0]
    coeffs_ls.append((x0, c0, c1, c2))       # store ascending for easy eval
coeffs_ls = np.array(coeffs_ls, dtype=float)

def eval_quarter_polyfit(x):
    # x in [0,0.25)
    j = np.minimum((x / h).astype(int), S-1)
    x0 = coeffs_ls[j,0]; c0 = coeffs_ls[j,1]; c1 = coeffs_ls[j,2]; c2 = coeffs_ls[j,3]
    return c0 + c1*x + c2*(x*x)

# ---------- Method 3:  Constrained least square -----

# Fit p(u) = y0 + B u + C u^2 on u∈[0,1], with p(1)=y1 constraint → B = (y1 - y0) - C
# Minimize ∫_0^1 [p(u) - g(u)]^2 du where g(u)=cos(2π(x0 + u h))
# We solve for C by least-squares on a dense grid (fast & robust).
def coeffs_endpoint_LS():
    coeffs = []
    ns = 200
    u = np.linspace(0, 1, ns)
    for j in range(S):
        x0 = nodes[j]; x1 = nodes[j+1]; hseg = x1 - x0
        y0 = cos_q(x0); y1 = cos_q(x1)
        g  = cos_q(x0 + u*hseg)
        # p(u) = y0 + ((y1-y0)-C) u + C u^2
        #       = y0 + (y1-y0)u + C(u^2 - u)
        # LS in C: minimize || (y0 + (y1-y0)u) + C*(u^2-u) - g ||_2
        phi = (u*u - u)
        rhs = g - (y0 + (y1 - y0)*u)
        C = np.dot(phi, rhs) / np.dot(phi, phi)      # 1D normal eq.
        B = (y1 - y0) - C
        coeffs.append((x0, y0, B, C))
    return np.array(coeffs)

def print_latex(coeffs, den_width=None):
    fraction = True if den_width != None else False
    for i in range(coeffs.shape[0]):
        x0=coeffs[i,0]
        y0=coeffs[i,1]
        B=coeffs[i,2]
        C=coeffs[i,3]
        if fraction:
            fy0=Fraction(y0).limit_denominator(10**DEN_WIDTH-1)
            fB=Fraction(B).limit_denominator(10**DEN_WIDTH-1)
            fC=Fraction(C).limit_denominator(10**DEN_WIDTH-1)
            y0o=fy0.numerator / fy0.denominator
            Bo=fB.numerator / fB.denominator
            Co=fC.numerator / fC.denominator
            print(f"\\frac{{{fy0.numerator}}}{{{fy0.denominator}}} - \\frac{{{np.abs(fB.numerator)}}}{{{fB.denominator}}}\\varphi - \\frac{{{np.abs(fC.numerator)}}}{{{fC.denominator}}}\\varphi^2, & \\varphi < \\frac{{{x0*4*S+1:.0f}}}{{{4*S}}},\\\\")
        else:
            print(f"{y0}{B}\\varphi{C}\\varphi^2, & \\varphi < \\frac{{{x0*4*S+1:.0f}}}{{{4*S}}},\\\\")

coeffs_eLS = coeffs_endpoint_LS()

def quantize(coeffs, shifts, multiplier_width=18, adder_width=24):
    c = coeffs.copy()
    bits = multiplier_width
    for i in range(2, c.shape[1]):
        c[:,i] = np.round(c[:,i]*2**(shifts[i-1]+bits)) / 2**(shifts[i-1]+bits)
    c[:,0] = np.round(c[:,0]*2**bits) / 2**+bits
    bits = adder_width
    c[:,1] = np.round(c[:,1]*2**(shifts[0]+bits-1)) / 2**(shifts[0]+bits-1)
    return c

def get_shifts(coeffs):
    shifts = []
    for i in range(1,coeffs.shape[1]):
        col = np.abs(coeffs[:,i])
        col = col[np.nonzero(col)]
        if col.size > 0:
            shifts.append(-np.max(np.log2(col)).astype(int))
        else:
            shifts.append(0)
    return np.array(shifts)

shifts = get_shifts(coeffs_eLS)

if QUANTIZE:
    coeffs_eLS = quantize(coeffs_eLS, shifts, MULTIPLIER_WIDTH, ADDER_WIDTH)

def eval_quarter_eLS(x):
    j = np.minimum((x / h).astype(int), S-1)
    x0 = coeffs_eLS[j,0]; A = coeffs_eLS[j,1]; B = coeffs_eLS[j,2]; C = coeffs_eLS[j,3]
    u = (x - x0)/h
    return A + B*u + C*u*u

# ---------- Common extension to full cycle ----------
def extend_full(eval_quarter_fn, x):
    # x is phase in [0,1)
    t = x
    y = np.empty_like(t)
    # Q1
    m = (t < 0.25)
    y[m] = eval_quarter_fn(t[m])
    # Q2
    m = (t >= 0.25) & (t < 0.5)
    y[m] = -eval_quarter_fn(0.5 - t[m])
    # Q3
    m = (t >= 0.5) & (t < 0.75)
    y[m] = -eval_quarter_fn(t[m] - 0.5)
    # Q4
    m = (t >= 0.75)
    y[m] = eval_quarter_fn(1.0 - t[m])
    return y

# Choose method
if METHOD == "polyfit":
    eval_q = eval_quarter_polyfit
elif METHOD == "interp":
    eval_q = eval_quarter_interp
elif METHOD == "constrained polyfit":
    eval_q = eval_quarter_eLS
else:
    raise ValueError("METHOD must be 'interp' or 'polyfit'")

def coeff_recto(coeffs):
    # Far-right quadratic with vertex pinned at y=0
    x0 = nodes[S-1]; x1 = nodes[S]     # x1 should be 0.25
    y0 = eval_q(x0)

    den = (x1 - x0)**2
    a = y0 / den
    b = -2.0 * a * x1
    c = a * x1 * x1

    return (x0, c, b, a)
re_x, re_c, re_b, re_a = coeff_recto(coeffs_eLS)
print(f"f({re_x}) = {re_c} + {re_b}x + {re_a}x^2")
# Generate waveform and a reference cosine
y = extend_full(eval_q, phase)
y_ref = np.cos(2*np.pi*phase)

# FFT (coherent): no window needed
Y = np.fft.rfft(y)/N
Y_ref = np.fft.rfft(y_ref)/N

freqs = np.fft.rfftfreq(N, 1/Fs)

# Extract first 15 harmonics magnitudes & dBc
k1 = np.argmin(np.abs(freqs - f0))  # should be the fundamental bin
fund = 2*np.abs(Y[k1])
fund_ref = 2*np.abs(Y_ref[k1])

rows = []
for k in range(1, 16):
    idx = k*k1
    if idx < len(Y):
        amp = 2*np.abs(Y[idx])
        dBc = 20*np.log10(amp / fund) if amp>0 else -np.inf
        rows.append({"harmonic": k, "freq (Hz)": freqs[idx], "amplitude": amp, "dBc": dBc})
    else:
        rows.append({"harmonic": k, "freq (Hz)": np.nan, "amplitude": 0.0, "dBc": -np.inf})

def print_table(rows: str):
    df = pl.DataFrame(rows)
    print("Fundamental amplitudes (approx vs pure cosine):", fund, fund_ref, "  bin:", k1)
    pl.Config.set_tbl_rows(-1)
    print(df)

print_table(rows)

# Plot magnitude spectrum up to Nyquist
def show_plot(Y, freqs, f0, Fs):
    plt.figure()
    mag = 20*np.log10(np.maximum(1e-20, 2*np.abs(Y)))
    plt.plot(freqs, mag)
    plt.xlim(0, Fs/2)
    plt.ylim(-160, 10)
    plt.title(f"Spectrum of {S}-segment quarter-wave cosine approx ({METHOD}, {f0} Hz @ {Fs/1000:.0f} kHz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dBFS rel. to 1.0 pk)")
    plt.tight_layout()
    plt.show()

print_latex(coeffs_eLS)

residual = Y - Y_ref
rms_res = np.sqrt(np.mean(np.abs(residual)**2))
rms_signal = np.sqrt(np.mean(np.abs(Y_ref)**2))
error_db = 20 * np.log10(rms_res / rms_signal)
print("Residual RMS:", rms_res)
print("Error (dB):", error_db)
print(f"Shift left by: {shifts}")
print("(constant coefficient can be added without shifts)")
ebits = [ADDER_WIDTH]
for i in [1,2]:
    ebits.append(shifts[i] + MULTIPLIER_WIDTH)
ebits = np.array(ebits)
print(f"Effective bits: {ebits}")
show_plot(Y, freqs, f0, Fs)
