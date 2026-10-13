#!/usr/bin/env python3
"""
Sensitivity analysis for ASSUMPTION-labelled parameters.
Varies each parameter by -25% and +25% from its nominal value on the reference case.
Outputs: sensitivity_reference.csv
"""
import sys, csv
sys.path.insert(0, '.')
import numpy as np
from pde_solver import (ThermalPDESolver, DIE_LX_DEFAULT, DIE_LY_DEFAULT,
                        DIE_THICKNESS_DEFAULT, H_EFF_DEFAULT, LEAKAGE_BETA_DEFAULT,
                        LEAKAGE_T0)
from sensor_generator import SIGMA_DEFAULT_C, QUANT_STEP_C, CALIB_OFFSET_RANGE_C

T_AMB = 35.0
REF_BLOCKS = [{
    'id': 'ref_cpu', 'type': 'CPU',
    'x_min': 0.003, 'x_max': 0.007,
    'y_min': 0.003, 'y_max': 0.007,
    'power_density_dynamic': 60.0,
    'power_density_leakage0': 3.0,
    'leakage_beta': LEAKAGE_BETA_DEFAULT,
}]

def solve_ref(t_die=DIE_THICKNESS_DEFAULT, h_eff=H_EFF_DEFAULT,
              beta=LEAKAGE_BETA_DEFAULT, p_leak0=3.0, p_dyn=60.0):
    blocks = [{
        'id': 'ref_cpu', 'type': 'CPU',
        'x_min': 0.003, 'x_max': 0.007,
        'y_min': 0.003, 'y_max': 0.007,
        'power_density_dynamic': p_dyn,
        'power_density_leakage0': p_leak0,
        'leakage_beta': beta,
    }]
    solver = ThermalPDESolver(grid_size=128, t_die=t_die, h_eff=h_eff)
    res = solver.solve(T_AMB, blocks)
    return res['peak_temp_C']

# Baseline
peak_baseline = solve_ref()
print(f"Baseline peak: {peak_baseline:.4f} °C")

# ASSUMPTION parameters (those labelled ASSUMPTION - NOT SOURCED or ASSUMPTION in PARAMETERS.md)
# We vary these; sourced parameters are not ASSUMPTION.
ASSUMPTION_PARAMS = {
    # param_name: (nominal_value, 'description', kwargs_key_or_None)
    'die_thickness':    (DIE_THICKNESS_DEFAULT, 'Die thickness t_die [m]', 't_die'),
    'h_eff':            (H_EFF_DEFAULT,          'Effective vertical heat-sink conductance h_eff [W/(m²K)]', 'h_eff'),
    'leakage_beta':     (LEAKAGE_BETA_DEFAULT,   'Leakage exponential coefficient beta [1/K]', 'beta'),
    'p_leak0':          (3.0,                    'Leakage power density at T0 [W/cm²]', 'p_leak0'),
    'sigma_sensor_C':   (SIGMA_DEFAULT_C,        'Sensor noise sigma [°C]', None),   # does not affect peak T
    'quant_step_C':     (QUANT_STEP_C,           'ADC quantization step [°C]', None),
    'calib_offset_range': (CALIB_OFFSET_RANGE_C,'Calibration offset range ±[°C]', None),
}

rows = []
for pname, (nominal, desc, kwarg_key) in ASSUMPTION_PARAMS.items():
    for direction, factor in [('-25%', 0.75), ('+25%', 1.25)]:
        varied = nominal * factor
        if kwarg_key is None:
            # sensor parameters do not change the solver: peak T unchanged
            peak = peak_baseline
        else:
            try:
                kw = {kwarg_key: varied}
                peak = solve_ref(**kw)
            except Exception as e:
                peak = float('nan')
                print(f"  ERROR for {pname} {direction}: {e}")
        delta = peak - peak_baseline
        rows.append({
            'parameter':      pname,
            'description':    desc,
            'nominal_value':  nominal,
            'unit':           '(see PARAMETERS.md)',
            'direction':      direction,
            'varied_value':   varied,
            'peak_temp_C':    round(peak, 6),
            'delta_peak_C':   round(delta, 6),
            'baseline_peak_C': round(peak_baseline, 6),
        })
        print(f"  {pname:25s}  {direction}  varied={varied:.6g}  peak={peak:.4f} °C  Δ={delta:+.4f} K")

out_path = 'py_pipeline/sensitivity_reference.csv'
fieldnames = ['parameter','description','nominal_value','unit','direction',
              'varied_value','peak_temp_C','delta_peak_C','baseline_peak_C']
with open(out_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
print(f"\nWrote {len(rows)} rows to {out_path}")
