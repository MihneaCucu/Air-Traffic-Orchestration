#!/usr/bin/env python3
import os
import sys

OUTPUT_DIR = "documentation_plots"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"✓ Created directory: {OUTPUT_DIR}/\n")

response = input("\nContinuă generarea graficelor? (y/n): ").lower()

if response != 'y':
    print("Anulat.")
    sys.exit(0)

try:
    print("Importing required modules...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    print("Modules imported successfully\n")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("\nPlease install required packages:")
    print("  pip install matplotlib numpy tensorboard")
    sys.exit(1)

try:
    from generate_documentation_plots import (
        plot_learning_curves,
        plot_final_performance,
        plot_training_stability,
        plot_convergence_speed,
        create_summary_figure,
        generate_latex_table
    )
    
    plot_learning_curves()
    plot_final_performance()
    plot_training_stability()
    plot_convergence_speed()
    create_summary_figure()
    generate_latex_table()
    
    print("\nAgent comparison plots completed!")
    
except Exception as e:
    print(f"Error generating agent comparison plots: {e}")
    print("Continuing with hyperparameter plots...")


try:
    from generate_hyperparameter_plots import (
        plot_hyperparameter_impact,
        plot_exploration_vs_exploitation,
        create_results_table_image
    )
    
    plot_hyperparameter_impact()
    plot_exploration_vs_exploitation()
    create_results_table_image()
    
    print("\nHyperparameter analysis plots completed!")
    
except Exception as e:
    print(f"Error generating hyperparameter plots: {e}")

if os.path.exists(OUTPUT_DIR):
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png') or f.endswith('.tex')])
    
    if files:
        print(f"\nTotal: {len(files)} fișiere în {OUTPUT_DIR}/\n")
        
        png_files = [f for f in files if f.endswith('.png')]
        tex_files = [f for f in files if f.endswith('.tex')]
        
        if png_files:
            print("Grafice (PNG):")
            for i, f in enumerate(png_files, 1):
                size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
                print(f"  {i:2d}. {f:<35} ({size:.1f} KB)")
        
        if tex_files:
            print("\n📝 Fișiere LaTeX:")
            for f in tex_files:
                print(f"  - {f}")
    else:
        print("\nNo files generated!")
else:
    print(f"\nDirectory {OUTPUT_DIR}/ not found!")

