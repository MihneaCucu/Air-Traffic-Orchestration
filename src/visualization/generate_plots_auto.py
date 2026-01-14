import os
import sys

OUTPUT_DIR = "documentation_plots"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    print("✓ Modules loaded\n")
except ImportError as e:
    print(f"Error: {e}")
    print("Install: pip install matplotlib numpy")
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
    print("Agent plots done!\n")
except Exception as e:
    print(f"Error: {e}\n")

try:
    from generate_hyperparameter_plots import (
        plot_hyperparameter_impact,
        plot_exploration_vs_exploitation,
        create_results_table_image
    )
    
    plot_hyperparameter_impact()
    plot_exploration_vs_exploitation()
    create_results_table_image()
    print("Hyperparameter plots done!\n")
except Exception as e:
    print(f"Error: {e}\n")

files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(('.png', '.tex'))]
print(f"Total: {len(files)} fișiere în {OUTPUT_DIR}/\n")

for f in sorted(files):
    if f.endswith('.png'):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"  ✓ {f:<40} ({size:.1f} KB)")
    else:
        print(f"  ✓ {f}")