import numpy as np
from fractions import Fraction

# ------------------ Controls ------------------
S = 32             # segments on [0.5,1)
NSAMP_LS = 200     # samples per segment for LS fit
DEN_WIDTH = 5      # limit_denominator(10**DEN_WIDTH-1) for nice LaTeX
QUANTIZE = False
# ----------------------------------------------

def f(x): return 1.0/x

nodes = np.linspace(0.5, 1.0, S+1)
h = 1/(2*S)

def fit_coeffs():
    """Return array of per-segment (x0, A, B, C) for p(u)=A+Bu+Cu^2, u∈[0,1]."""
    u = np.linspace(0.0, 1.0, NSAMP_LS)
    coeffs = []
    for j in range(S):
        x0, x1 = nodes[j], nodes[j+1]
        y0, y1 = f(x0), f(x1)
        g = f(x0 + u*(x1-x0))

        # Constrained LS in one unknown (C); B=(y1-y0)-C, A=y0
        phi = (u*u - u)
        rhs = g - (y0 + (y1 - y0)*u)
        C = np.dot(phi, rhs) / np.dot(phi, phi)
        B = (y1 - y0) - C
        A = y0

        if QUANTIZE:
            q = 10**DEN_WIDTH - 1
            Af = Fraction(A).limit_denominator(q)
            Bf = Fraction(B).limit_denominator(q)
            Cf = Fraction(C).limit_denominator(q)
            A = Af.numerator / Af.denominator
            B = Bf.numerator / Bf.denominator
            C = Cf.numerator / Cf.denominator
            coeffs.append((x0, A, B, C, Af, Bf, Cf))
        else:
            coeffs.append((x0, A, B, C, None, None, None))
    return coeffs

coeffs = fit_coeffs()

def print_piecewise_latex():
    r"""
    Prints something like:
      \frac{1}{x} \approx
      \begin{cases}
        A_0 + B_0 \varphi + C_0 \varphi^2, & x < 1 + \frac{1}{32},\\
        ...
      \end{cases}
    where \varphi = \frac{x-x_0}{h}.
    """
    print(r"\[")
    print(r"\frac{1}{x}\approx")
    print(r"\begin{cases}")
    for j,(x0,A,B,C,Af,Bf,Cf) in enumerate(coeffs):
        # pretty fractions when available
        def fmt(flt, Frac):
            if Frac is None:
                return f"{flt:.10g}"
            n,d = Frac.numerator, Frac.denominator
            if d==1: return f"{n}"
            return rf"\frac{{{n}}}{{{d}}}"
        Atex = fmt(A, coeffs[j][4])
        Btex = fmt(B, coeffs[j][5])
        Ctex = fmt(C, coeffs[j][6])

        # end condition x < 1 + (j+1)/S
        rhs_num = j+1+S
        rhs_den = S*2

        line = rf"{Atex} + {Btex}\,\varphi + {Ctex}\,\varphi^2, & x < \frac{{{rhs_num}}}{{{rhs_den}}},\\"
        print(line)
    # final "otherwise" row (exactly at x=2 falls into last segment)
    print(r"\text{otherwise.}")
    print(r"\end{cases}")
    print(r"\]")
    print()
    print(r"where $\varphi=\dfrac{x-x_0}{h}$, with $x_0=1+\dfrac{j}{32}$ and $h=\dfrac{1}{32}$ for segment $j=0,\dots,31$.")

print_piecewise_latex()

# Optional: quick error stats on [1,2]
xs = np.linspace(0.5, 1.0, 20001)
def eval_inv(x):
    x = np.minimum(np.maximum(x, 0.5), np.nextafter(1.0, 0.5))
    j = np.minimum(((x-0.5)/h).astype(int), S-1)
    x0,A,B,C,_,_,_ = list(zip(*[coeffs[k] for k in j]))
    x0 = np.array(x0); A=np.array(A); B=np.array(B); C=np.array(C)
    u = (x - x0)/h
    return A + B*u + C*u*u

y_true = 1.0/xs
y_apx  = eval_inv(xs)
abs_err = np.abs(y_true - y_apx)
rel_err = abs_err / y_true
print("max abs err: {} ({})".format(np.max(abs_err), np.log2(np.max(abs_err))))
print("max rel err:", np.max(rel_err))
print("rms rel err:", np.sqrt(np.mean(rel_err**2)))
