"""
SAC (Soft Actor-Critic) Package
================================

All SAC files and models are organized in this folder.

Modules:
--------
- sac_agent.py: SAC agent implementation
- train_sac.py: Training script
- eval_sac.py: Evaluation script
- test_sac.py: Testing script
- visualize_sac.py: Visualization script
- test_sac_complete.py: Complete component testing

Models:
-------
- sac_atc.pth: Final model
- sac_atc_best.pth: Best model (recommended)
- sac_checkpoints/: Periodic checkpoints

Usage:
------
cd SAC
python eval_sac.py --best
python visualize_sac.py --best
python train_sac.py --quick
"""

__version__ = '1.0.0'
__all__ = ['sac_agent', 'train_sac', 'eval_sac', 'test_sac', 'visualize_sac']

