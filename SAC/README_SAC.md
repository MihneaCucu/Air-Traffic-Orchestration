# SAC (Soft Actor-Critic) pentru ATC Environment

## 📋 Cuprins
- [Ce este SAC?](#ce-este-sac)
- [Structura Fișierelor](#structura-fișierelor)
- [Instalare & Setup](#instalare--setup)
- [Cum să Rulezi SAC](#cum-să-rulezi-sac)
- [Modele Antrenate](#modele-antrenate)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

---

## 🤖 Ce este SAC?

**SAC (Soft Actor-Critic)** este un algoritm de reinforcement learning off-policy care:
- ✅ Maximizează reward-ul ȘI entropia (explorare inteligentă)
- ✅ Este foarte stable la training
- ✅ Funcționează bine pentru continuous și discrete action spaces
- ✅ Are automatic temperature tuning

### Avantajele SAC vs alte algoritme:
| Caracteristică | SAC | PPO | DQN |
|----------------|-----|-----|-----|
| Sample efficiency | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Stabilitate | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Explorare | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Viteză training | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 📁 Structura Fișierelor

```
Project/
├── sac_agent.py          # Implementarea SAC (agent + rețele)
├── train_sac.py          # Script pentru antrenament
├── eval_sac.py           # Script pentru evaluare
├── test_sac.py           # Script pentru testare componente
├── visualize_sac.py      # Script pentru vizualizare cu rendering
├── atc_env.py            # Environment-ul ATC
└── models/               # Directorul cu modele
    ├── sac_atc.pth      # Model final (ultima versiune)
    ├── sac_atc_best.pth # Best model (cel mai bun reward)
    └── sac_checkpoints/ # Checkpoint-uri la fiecare 50k steps
        ├── sac_atc_50000.pth
        ├── sac_atc_100000.pth
        ├── ...
        └── sac_atc_500000.pth
```

---

## 🔧 Instalare & Setup

### 1. Activează Virtual Environment

```bash
cd /Users/mihneacucu/Documents/RL/Project
source venv/bin/activate
```

### 2. Verifică Dependențele

Asigură-te că ai instalate:
```bash
pip install torch numpy pygame gymnasium
```

### 3. Verifică că Totul Funcționează

```bash
python test_sac.py
```

Dacă vezi `✅ ALL TESTS PASSED!`, ești gata!

---

## 🚀 Cum să Rulezi SAC

### 📊 **1. EVALUARE (Recomandat să începi cu asta!)**

#### Evaluează Best Model (recomandat)
```bash
python eval_sac.py --best
```

Acesta este **cel mai bun model** salvat în timpul antrenamentului.

#### Evaluează Model Final
```bash
python eval_sac.py
```

#### Evaluare Extinsă (mai multe episoade)
```bash
python eval_sac.py --best --episodes 50
```

#### Compară Toate Modelele
```bash
python eval_sac.py --compare
```

Compară modelul final, best model și toate checkpoint-urile!

#### Evaluare cu Render (Vezi agentul în acțiune)
```bash
python eval_sac.py --best --render --episodes 5
```

**Output-ul arată:**
```
======================================================================
EVALUARE SAC AGENT
======================================================================
Model: models/sac_atc_best.pth
Device: cpu
...
======================================================================
📊 REZULTATE EVALUARE
======================================================================
Episoade evaluate: 100

Reward Statistics:
  Mean Reward:    156.34 ± 45.21
  Min Reward:     -23.45
  Max Reward:     287.90

Episode Statistics:
  Mean Length:    198.45 ± 23.12
  Success Rate:    78.0% (78/100)
======================================================================
```

---

### 🎬 **2. VIZUALIZARE (Vezi agentul cum joacă)**

#### Vizualizare Best Model
```bash
python visualize_sac.py --best
```

Deschide o fereastră pygame și vezi agentul SAC controlând avioanele în timp real!

#### Vizualizare cu Mai Multe Episoade
```bash
python visualize_sac.py --best --episodes 10
```

#### Vizualizare în Slow Motion (pentru debugging)
```bash
python visualize_sac.py --best --speed slow
```

#### Opțiuni de viteză:
- `--speed slow` - 5 FPS (pentru analiză detaliată)
- `--speed normal` - 10 FPS (default)
- `--speed fast` - 20 FPS (rapid)

**Output:**
```
======================================================================
VIZUALIZARE AGENT SAC
======================================================================
Episoade: 5
Render speed: normal (10 FPS)
======================================================================

⏸️  Închide fereastra pentru a opri vizualizarea

======================================================================
📺 EPISOD 1/5
======================================================================
  Step 50: Reward=2.50, Total Score=125.30
  Step 100: Reward=3.20, Total Score=245.80

✓ SUCCESS
  Final Score: 267.45
  Steps: 127
======================================================================
```

---

### 🏋️ **3. ANTRENAMENT (Dacă vrei să antrenezi din nou)**

⚠️ **ATENȚIE**: Deja ai modele antrenate! Antrenamentul va suprascrie modelele existente!

#### Quick Training (testare rapidă, ~10-15 min)
```bash
python train_sac.py --quick
```
- 100,000 steps
- Perfect pentru testare rapidă

#### Full Training (recomandat, ~1-2 ore)
```bash
python train_sac.py
```
- 500,000 steps
- Balansat între timp și performanță

#### Long Training (pentru cei mai buni rezultate, ~3-4 ore)
```bash
python train_sac.py --long
```
- 1,000,000 steps
- Cea mai bună performanță

#### Training Custom
```bash
python train_sac.py --timesteps 750000
```

**Output în timpul training-ului:**
```
======================================================================
ANTRENAMENT SAC PENTRU ATC ENVIRONMENT
======================================================================
Device: cpu
Total timesteps: 500,000
Learning starts: 10,000
Batch size: 256
Buffer size: 1,000,000
======================================================================

🚀 Starting training...
📊 Logs: atc_logs/sac_training_20260114-123456.log
----------------------------------------------------------------------
Episode   10 | Step    2847 | Reward:  -45.23 | Avg(10):  -52.34 | Len: 284 | Q1Loss: 0.1234 | PolLoss: 0.0456
Episode   20 | Step    5821 | Reward:   12.45 | Avg(10):   -8.91 | Len: 297 | Q1Loss: 0.0987 | PolLoss: 0.0389

======================================================================
📊 EVALUATION at step 10,000
----------------------------------------------------------------------
Mean Reward: 45.67
Mean Length: 189.34
Success Rate: 34.0%
======================================================================

🏆 New best model! Saved to models/sac_atc_best.pth
...
```

---

## 📦 Modele Antrenate

### Diferența între Modele

#### `sac_atc.pth` - Model Final
- ✅ Ultima versiune salvată la sfârșitul antrenamentului
- ⚠️ Poate să **nu fie** cea mai bună versiune
- 📍 Folosește doar dacă vrei să continui training-ul

#### `sac_atc_best.pth` - Best Model ⭐ **RECOMANDAT**
- ✅ Modelul cu **cel mai mare reward mediu** în evaluare
- ✅ Salvat automat când agentul atinge record nou
- ✅ **Cel mai bun pentru evaluare și deployment**
- 🎯 **Folosește întotdeauna acesta pentru demonstrații!**

#### `sac_checkpoints/sac_atc_*.pth` - Checkpoint-uri
- ✅ Salvate la fiecare 50,000 steps
- ✅ Utile pentru:
  - Comparare progres
  - Recovery dacă training-ul se întrerupe
  - Analiză evoluție agent

### Cum Verific Ce Modele Am?

```bash
# Vezi modelele principale
ls -lh models/sac_atc*.pth

# Vezi toate checkpoint-urile
ls -lh models/sac_checkpoints/

# Sau folosește eval pentru comparație
python eval_sac.py --compare
```

---

## 🧪 Testare

### Test Rapid Componente
```bash
python test_sac.py
```

Verifică:
- ✅ Environment funcționează
- ✅ Replay buffer funcționează
- ✅ Agent se poate crea
- ✅ Training loop funcționează

### Test Quick Training (1000 steps)
```bash
python test_sac.py
```

---

## 🔧 Troubleshooting

### Problema: Scriptul se blochează la încărcare

**Cauză**: PyTorch are probleme de threading pe macOS cu Python 3.13

**Soluție**: Toate scripturile au fost patches cu:
```python
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)
```

Dacă tot nu merge:
```bash
# Reinstalează PyTorch
pip uninstall torch
pip install torch --no-cache-dir

# SAU downgrade la Python 3.11
conda create -n rl-py311 python=3.11
conda activate rl-py311
pip install torch numpy pygame gymnasium
```

### Problema: "Model nu a fost găsit"

**Verifică**:
```bash
ls -la models/sac_atc*.pth
```

**Dacă lipsesc**: Folosește un checkpoint:
```bash
cp models/sac_checkpoints/sac_atc_500000.pth models/sac_atc_best.pth
```

### Problema: Pygame warning despre pkg_resources

**Nu este o problemă!** Este doar un warning. Scriptul va funcționa normal.

### Problema: Import error "No module named 'sac_agent'"

**Soluție**: Asigură-te că rulezi din directorul corect:
```bash
cd /Users/mihneacucu/Documents/RL/Project
python eval_sac.py --best
```

---

## ❓ FAQ

### Q: Am nevoie să antrenez din nou SAC?
**A**: **NU!** Dacă ai deja `sac_atc_best.pth`, poți evalua și vizualiza direct:
```bash
python eval_sac.py --best
python visualize_sac.py --best
```

### Q: Care model să folosesc pentru evaluare?
**A**: **Întotdeauna `--best`**:
```bash
python eval_sac.py --best
python visualize_sac.py --best
```

### Q: Cât timp durează antrenamentul?
**A**: 
- Quick (100k): ~10-15 minute
- Full (500k): ~1-2 ore
- Long (1M): ~3-4 ore

### Q: Cum compar SAC cu PPO/DQN?
**A**: Folosește script-ul de comparație:
```bash
python compare_all_agents.py
```

### Q: Pot relua antrenamentul de unde a rămas?
**A**: Da, modifică `train_sac.py` să încarce un checkpoint:
```python
agent.load("models/sac_checkpoints/sac_atc_500000.pth")
```

### Q: SAC este mai bun decât PPO?
**A**: Depinde de task:
- **SAC** → Mai sample efficient, explorare mai bună
- **PPO** → Mai simplu, mai stable pentru unele taskuri
- Rulează `compare_all_agents.py` pentru a compara pe task-ul tău!

### Q: Cum văd progresul în timp real?
**A**: Logurile se salvează în `atc_logs/`:
```bash
# Vezi ultimele linii
tail -f atc_logs/sac_training_*.log

# SAU folosește TensorBoard (dacă e configurat)
tensorboard --logdir=atc_logs
```

---

## 📚 Resurse Suplimentare

### Documentație
- `SAC_CHEATSHEET.md` - Quick reference pentru comenzi
- `sac_agent.py` - Codul sursă cu comentarii detaliate

### Papers
- [Soft Actor-Critic (Original)](https://arxiv.org/abs/1801.01290)
- [SAC for Discrete Actions](https://arxiv.org/abs/1910.07207)

### Comenzi Utile Rapid

```bash
# Evaluare rapidă
python eval_sac.py --best --episodes 20

# Vizualizare
python visualize_sac.py --best

# Comparație modele
python eval_sac.py --compare

# Training nou (doar dacă e necesar!)
python train_sac.py --quick

# Test componente
python test_sac.py
```

---

## 🎯 Quick Start pentru Începători

**Ești nou? Urmează acești pași:**

1. **Activează environment-ul**:
   ```bash
   cd /Users/mihneacucu/Documents/RL/Project
   source venv/bin/activate
   ```

2. **Evaluează modelul existent**:
   ```bash
   python eval_sac.py --best
   ```

3. **Vezi agentul în acțiune**:
   ```bash
   python visualize_sac.py --best --episodes 3
   ```

4. **Compară cu alte modele** (opțional):
   ```bash
   python eval_sac.py --compare
   ```

**Gata! Acum știi cum funcționează SAC pe environment-ul tău!** 🚀

---

## 📧 Suport

Pentru întrebări sau probleme:
1. Verifică secțiunea [Troubleshooting](#troubleshooting)
2. Verifică FAQ-ul
3. Rulează `python test_sac.py` pentru diagnosticare

---

**Happy Training! 🎉**

