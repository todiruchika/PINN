#!/usr/bin/env python3
"""
Acceptance Test T6:
Running the generator twice with the same seed gives identical output.
Reports max absolute difference.
"""
import os, sys, shutil
import numpy as np

# Ensure writable matplotlib cache dir
os.environ['MPLCONFIGDIR'] = '/tmp/mpl'
sys.path.insert(0, 'py_pipeline')

from dataset_generator import MasterDatasetGenerator

def run_test():
    dir1 = '/tmp/test_t6_run1'
    dir2 = '/tmp/test_t6_run2'
    if os.path.exists(dir1): shutil.rmtree(dir1)
    if os.path.exists(dir2): shutil.rmtree(dir2)

    seed = 12345
    T_amb = 33.5
    fp_id = 7
    sample_idx = 1

    # Generator 1
    gen1 = MasterDatasetGenerator(output_dir=dir1, grid_size=128, use_fine_mesh=False, global_seed=seed)
    gen1.prepare_directories(['test'])
    meta1 = gen1.generate_single_sample(sample_idx, 'test', T_amb, fp_id)
    T1 = np.load(os.path.join(dir1, 'test', 'matrices', f'thermal_{sample_idx:05d}.npy'))

    # Generator 2
    gen2 = MasterDatasetGenerator(output_dir=dir2, grid_size=128, use_fine_mesh=False, global_seed=seed)
    gen2.prepare_directories(['test'])
    meta2 = gen2.generate_single_sample(sample_idx, 'test', T_amb, fp_id)
    T2 = np.load(os.path.join(dir2, 'test', 'matrices', f'thermal_{sample_idx:05d}.npy'))

    max_diff = float(np.max(np.abs(T1 - T2)))
    print(f"T6 (Coarse 128x128): max absolute difference = {max_diff:.10e}")
    assert max_diff == 0.0, f"Outputs not identical: diff={max_diff}"

    # Also test fine-mesh reference solver
    gen1_fine = MasterDatasetGenerator(output_dir=dir1, grid_size=128, use_fine_mesh=True, global_seed=seed)
    meta1_fine = gen1_fine.generate_single_sample(sample_idx, 'test', T_amb, fp_id)
    T1_fine = np.load(os.path.join(dir1, 'test', 'matrices', f'thermal_{sample_idx:05d}.npy'))

    gen2_fine = MasterDatasetGenerator(output_dir=dir2, grid_size=128, use_fine_mesh=True, global_seed=seed)
    meta2_fine = gen2_fine.generate_single_sample(sample_idx, 'test', T_amb, fp_id)
    T2_fine = np.load(os.path.join(dir2, 'test', 'matrices', f'thermal_{sample_idx:05d}.npy'))

    max_diff_fine = float(np.max(np.abs(T1_fine - T2_fine)))
    print(f"T6 (FineMesh 512->128): max absolute difference = {max_diff_fine:.10e}")
    assert max_diff_fine == 0.0, f"Fine mesh outputs not identical: diff={max_diff_fine}"

    print(f"ACCEPTANCE TEST T6: PASS (max absolute difference = {max(max_diff, max_diff_fine):.10e})")

    # Cleanup
    shutil.rmtree(dir1)
    shutil.rmtree(dir2)

if __name__ == '__main__':
    run_test()
