DSPMath
=======

The Synthesizable Logic Microsystems™ DSPMath library is a SpinalHDL library of
mathematical functions for digital signal processing under the
io.github.slmicro.dsp namespace.

The DSPMath library provides synthesizable fixed-point approximations of common
math and DSP functions. These are optimized for FPGAs where multiplier
precision and LUT depth are at a premium.

## Quadratics

Many functions are implemented as multi-segment quadratics using a constrained
least-squares fit.  For these constraints, a segment's f(0)=f*(length) where
f*() is the prior segment.  This means each segment avoids a discontinuity, but
still retains a slope discontinuity.

The segments are defined by tables of coefficients normalized from
x∈[0,length).  This allows the top bits to be used as a segment index to select
coefficients, and the lower bits to be shifted left before sending through
multipliers.  As such, several bits of precision are gained over the input size
for the multiplier:  a 32-segment quadratic uses 5 bits for the index, and so
an 18x18 multiplier can operate with 23 bits of precision for x.

Coefficients are also automatically shifted based on the number of leading
zeroes in the largest coefficient.  For many functions, this increases as the
number of segments increases, making the added precision meaningful and
allowing high precision on lower-cost FPGAs without consuming a large number of
hardware multipliers.

Small coefficient tables are also synthesized as efficient logic.  A table of
16 coefficiets fits in a single row of 4-input LUTs; 32 coefficients can use a
single half-wide row of 6-input, 2-output LUTs, and 64 coefficients use a full
row of 6-input, 2-output LUTs.  Hierarchy is needed for bigger coefficient
tables, for example 32 segment quadratics require a 2:1 mux to switch between
two banks of 16 coefficients when implemented on 4-input LUTs.

These piecewise quadratic functions provide significant precision with few
segments.  For example, maximum absolute error:

| Function | 16-segment       | 32-segment       | 64-segment       |
| -------- | ---------------- | ---------------- | ---------------- |
| 1/x      | 2.1 e-5 (15.6 b) | 2.8 e-6 (18.5 b) | 3.6 e-7 (21.4 b) |
| exp2(x)  | 1.0 e-5 (16.6 b) | 1.3 e-6 (19.6 b) | 1.6 e-7 (22.6 b) |
| cos(θ)   | −105 dB          | −123 dB          | −140 dB          |

Maximum relative error:

| Function | 16-segment | 32-segment | 64-segment |
|---|---|---|---|
| 1/x | 1.1e-5 |1.4 e-6 |1.8e-7|
| exp2(x) | 5.1e-6 | 6.4e-7 | 8.1e-8 |

This level of precision is sufficient for most audio applications, such as
calculating frequency from cents, phase modulation, or PolyBLEP kernels.
