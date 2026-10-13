#!/usr/bin/env python3
"""
Scientific Visualization Suite for PINN Thermal Datasets
Part of the PINN Thermal Dataset Generation Pipeline.

Features:
  1. Plot standard full thermal map with coordinates and colorbars
  2. Overlay floorplan hardware blocks and sparse sensor markers
  3. Display side-by-side comparisons of multiple chip scenarios
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from typing import List, Dict, Any

class ThermalVisualizer:
    @staticmethod
    def plot_single_sample(matrix_path: str, sensor_path: str = None, metadata_row: Dict[str, Any] = None, save_path: str = None):
        """
        Loads and plots a single thermal map, overlaying layout boundaries and sensor dots.
        """
        # Load temperature grid
        if not os.path.exists(matrix_path):
            raise FileNotFoundError(f"Matrix file not found: {matrix_path}")
        
        matrix = np.load(matrix_path)
        grid_size = matrix.shape[0]
        
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        
        # Die dimensions (default 12 mm x 12 mm)
        Lx = float(metadata_row.get("die_L_x_m", 0.012)) if metadata_row else 0.012
        Ly = float(metadata_row.get("die_L_y_m", 0.012)) if metadata_row else 0.012
        Lx_mm, Ly_mm = Lx * 1e3, Ly * 1e3

        # Render Heatmap with Inferno colormap in mm coordinates
        im = ax.imshow(matrix, cmap='inferno', origin='lower', extent=[0, Lx_mm, 0, Ly_mm], vmin=matrix.min(), vmax=matrix.max())
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Temperature (°C)", rotation=270, labelpad=15)
        
        # Overlay component blocks if metadata provided
        if metadata_row is not None:
            if "blocks_json" in metadata_row:
                blocks = json.loads(metadata_row["blocks_json"])
                for b in blocks:
                    x_min_mm = b["x_min"] * 1e3
                    y_min_mm = b["y_min"] * 1e3
                    w_mm = (b["x_max"] - b["x_min"]) * 1e3
                    h_mm = (b["y_max"] - b["y_min"]) * 1e3
                    p_dyn = b.get("power_density_dynamic", 0.0)
                    rect = patches.Rectangle(
                        (x_min_mm, y_min_mm), w_mm, h_mm,
                        linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle='--', alpha=0.8
                    )
                    ax.add_patch(rect)
                    ax.text(x_min_mm + 0.1, y_min_mm + 0.1, f"{b['type']} {p_dyn:.0f}W/cm²",
                            color='cyan', fontsize=6, weight='bold', alpha=0.9)
            elif "hotspot_positions" in metadata_row:
                blocks = json.loads(metadata_row["hotspot_positions"])
                powers = json.loads(metadata_row["hotspot_power_values"])
                for b, p in zip(blocks, powers):
                    x_min, y_min, x_max, y_max = b
                    rect = patches.Rectangle(
                        (x_min * Lx_mm, y_min * Ly_mm), (x_max - x_min) * Lx_mm, (y_max - y_min) * Ly_mm,
                        linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle='--', alpha=0.8
                    )
                    ax.add_patch(rect)
                
        # Overlay sparse sensor coordinates
        if sensor_path is not None and os.path.exists(sensor_path):
            with open(sensor_path, "r") as f:
                sensor_data = json.load(f)
            
            positions = np.array(sensor_data["positions"])
            # convert positions to mm if in metres (max < 0.1)
            if positions.size > 0 and positions.max() < 0.1:
                pos_plot = positions * 1e3
            else:
                pos_plot = positions * Lx_mm if positions.max() <= 1.0 else positions
                
            temps = np.array(sensor_data.get("temperatures_noisy", sensor_data.get("temperatures", [])))
            
            # Scatter plot coordinates
            ax.scatter(pos_plot[:, 0], pos_plot[:, 1], color='lime', edgecolors='black', s=50, marker='o', label='Sensors', zorder=5)
            # Annotate with temperature readings on top
            for p, t in zip(pos_plot, temps):
                ax.text(p[0] + 0.15, p[1] + 0.15, f"{t:.1f}°C", color='lime', fontsize=7, weight='bold',
                        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2', edgecolor='none'), zorder=6)
                
        # Set Labels and Grid details
        sample_title = metadata_row["sample_id"] if metadata_row is not None else os.path.basename(matrix_path)
        ax.set_title(f"Thermal Map Solution - {sample_title}", fontsize=11, weight='bold', pad=10)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_xlim(0, Lx_mm)
        ax.set_ylim(0, Ly_mm)
        ax.grid(True, which='both', color='white', linestyle=':', alpha=0.25)
        
        if sensor_path is not None and os.path.exists(sensor_path):
            ax.legend(loc='upper right', framealpha=0.8, facecolor='black', edgecolor='none', labelcolor='white')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"[Visualizer] Figure successfully saved to {save_path}", flush=True)
        else:
            plt.show()
            
        plt.close()

    @staticmethod
    def plot_comparison(matrix_paths: List[str], titles: List[str], save_path: str = None):
        """
        Draws side-by-side subplots comparing multiple solution matrices.
        """
        num_mats = len(matrix_paths)
        fig, axes = plt.subplots(1, num_mats, figsize=(4 * num_mats, 4.5), dpi=100)
        
        if num_mats == 1:
            axes = [axes]
            
        for i, (path, title) in enumerate(zip(matrix_paths, titles)):
            if not os.path.exists(path):
                continue
            matrix = np.load(path)
            im = axes[i].imshow(matrix, cmap='inferno', origin='lower', extent=[0, 1, 0, 1])
            axes[i].set_title(title, fontsize=10, weight='bold')
            axes[i].set_xlim(0, 1)
            axes[i].set_ylim(0, 1)
            axes[i].axis('off')
            
            # Universal colorbar
            cbar = fig.colorbar(im, ax=axes[i], shrink=0.7)
            cbar.set_label("°C", size=9)
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[Visualizer] Comparison fig saved to {save_path}", flush=True)
        else:
            plt.show()
            
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thermal Map Mathematical Visualization Suite")
    parser.add_argument("--matrix", type=str, required=True, help="Path to solution .npy file")
    parser.add_argument("--sensors", type=str, default=None, help="Path to sensors .json file")
    parser.add_argument("--save", type=str, default=None, help="Save to specific image path (.jpg or .png)")
    parser.add_argument("--meta-row-json", type=str, default=None,
                        help="Single row of metadata in JSON format (passed string) detailing active hotspot segments")
    
    args = parser.parse_args()
    
    meta_dict = None
    if args.meta_row_json:
        meta_dict = json.loads(args.meta_row_json)
        
    ThermalVisualizer.plot_single_sample(
        matrix_path=args.matrix,
        sensor_path=args.sensors,
        metadata_row=meta_dict,
        save_path=args.save
    )
