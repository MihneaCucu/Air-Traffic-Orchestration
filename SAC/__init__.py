"""
SAC (Soft Actor-Critic) Package
================================

Toate fișierele și modelele SAC sunt organizate în acest folder.

Modules:
--------
- sac_agent.py: Implementarea SAC agent
- train_sac.py: Script antrenament
- eval_sac.py: Script evaluare
- test_sac.py: Script testare
- visualize_sac.py: Script vizualizare
- test_sac_complete.py: Test complet toate componente

Models:
-------
- sac_atc.pth: Model final
- sac_atc_best.pth: Best model (recomandat)
- sac_checkpoints/: Checkpoint-uri periodice

Usage:
------
cd SAC
python eval_sac.py --best
python visualize_sac.py --best
python train_sac.py --quick
"""

__version__ = '1.0.0'
__all__ = ['sac_agent', 'train_sac', 'eval_sac', 'test_sac', 'visualize_sac']

