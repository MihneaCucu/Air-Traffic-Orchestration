# Air Traffic Control Orchestration cu Reinforcement Learning

Acest proiect implementează și compară diverși agenți de Reinforcement Learning (PPO, Custom DQN) pentru problema dirijării traficului aerian.

## Structură
- `atc_env.py`: Mediul de simulare (Gymnasium).
- `custom_dqn_agent.py`: Implementarea manuală a algoritmului DQN.
- `train.py`: Script pentru antrenarea agenților folosind Stable Baselines3 (PPO, DQN).
- `train_custom_dqn.py`: Script pentru antrenarea agentului Custom DQN.
- `compare_agents.py`: Script pentru evaluarea comparativă a agenților antrenați.
- `visualize_agent.py`: Vizualizare grafică a comportamentului unui agent.

## Pași de rulare

### 1. Antrenare
Poți antrena agenții separat.

**PPO (Stable Baselines3):**
Modifică `train.py` dacă vrei doar PPO și rulează:
```bash
python train.py
```

**Custom DQN (Implementare proprie):**
Acest script antrenează implementarea manuală de DQN.
```bash
python train_custom_dqn.py
```
*Notă: Ajustați `total_timesteps` în scripturi pentru antrenamente mai lungi (recomandat > 500k pași).*

### 2. Monitorizare Antrenament (TensorBoard)
Pentru a vedea graficele de reward și loss în timp real:
```bash
tensorboard --logdir atc_logs
```
Deschideți linkul generat în browser (ex: `http://localhost:6006`).

### 3. Comparare Rezultate
După ce ai antrenat modelele (se vor salva în folderul `models/`), poți rula scriptul de comparație:
```bash
python compare_agents.py
```
Acesta va rula fiecare agent pentru un număr de episoade, va calcula media recompenselor și va genera un grafic `rezultate_comparative.png`.

### 4. Vizualizare Agent
Pentru a vedea cum se comportă agentul în mediul vizual:
```bash
python visualize_agent.py
```
*(Asigură-te că în visualize_agent.py este încărcat modelul corect, ex: `CustomDQN` sau `PPO`).*

## Documentație
Pentru documentație poți folosi:
- Graficele din TensorBoard (pentru curbele de învățare).
- Imaginea `rezultate_comparative.png` (pentru performanța finală).
- Codul din `custom_dqn_agent.py` ca exemplu de implementare a algoritmului.