import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

class ATC2DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super(ATC2DEnv, self).__init__()
        
        # --- CONFIGURAȚIE ---
        self.area_size = 100.0  # Spațiul aerian 100x100 km
        self.num_intruders = 3  # Număr fix de intrusi pentru simplitatea inputului
        self.min_separation = 5.0  # 5 unități distanță orizontală
        self.speed = 1.0  # Viteză constantă pe pas
        self.turn_rate = np.deg2rad(15) # 15 grade pe step
        
        # --- ACTION SPACE ---
        # 0: Menține, 1: Stânga, 2: Dreapta, 3: Urcă, 4: Coboară
        self.action_space = spaces.Discrete(5)

        # --- OBSERVATION SPACE ---
        # State: [Own_x, Own_y, Own_alt, Own_heading, Target_x, Target_y] + 
        #        N * [Intruder_x, Intruder_y, Intruder_alt, Intruder_vx, Intruder_vy]
        # Total elemente: 6 + (5 * num_intruders)
        obs_size = 7 + (5 * self.num_intruders) 
        
        # Definim limitele (normalizate sau brute, aici folosim brute pt claritate)
        low = np.full((obs_size,), -np.inf)
        high = np.full((obs_size,), np.inf)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Rendering
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.font = None


    def _get_obs(self):
        # Normalizăm valorile între 0 și 1
        scale = self.area_size
        
        obs = [
            self.agent_pos[0] / scale, 
            self.agent_pos[1] / scale, 
            self.agent_alt / 2.0,       # Altitudinea e 0, 1 sau 2 -> devine 0.0, 0.5, 1.0
            math.cos(self.agent_heading), # Sin/Cos sunt deja intre -1 si 1, e perfect
            math.sin(self.agent_heading), # E mai bine sa dai sin/cos decat unghiul brut
            self.target_pos[0] / scale, 
            self.target_pos[1] / scale
        ]
        
        for intr in self.intruders:
            obs.extend([
                intr['pos'][0] / scale, 
                intr['pos'][1] / scale, 
                intr['alt'] / 2.0, 
                intr['vx'], # Vitezele sunt mici (-1 la 1), e ok
                intr['vy']
            ])
            
        return np.array(obs, dtype=np.float32)
    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Agentul și Ținta rămân la fel
        self.agent_pos = np.array([10.0, 10.0], dtype=np.float32)
        self.agent_heading = np.deg2rad(45)
        self.agent_alt = 1
        self.target_pos = np.array([90.0, 90.0], dtype=np.float32)
        
        # 2. Intrusii - Adăugăm logică de siguranță
        self.intruders = []
        for _ in range(self.num_intruders):
            valid_position = False
            while not valid_position:
                # Generăm o poziție
                pos = np.random.uniform(0, 100, size=2)
                
                # Verificăm distanța față de agent
                dist_to_agent = np.linalg.norm(pos - self.agent_pos)
                
                # Acceptăm poziția doar dacă e la cel puțin 25 unități distanță
                if dist_to_agent > 25.0:
                    valid_position = True
            
            # Restul codului rămâne la fel...
            alt = np.random.randint(0, 3)
            angle = np.random.uniform(0, 2 * np.pi)
            vx = math.cos(angle) * self.speed * 0.8
            vy = math.sin(angle) * self.speed * 0.8
            
            self.intruders.append({'pos': pos, 'alt': alt, 'vx': vx, 'vy': vy})

        self.step_count = 0
        return self._get_obs(), {}
    

    def step(self, action):
        self.step_count += 1
        reward = -0.2 # Penalizare mică pentru timp (combustibil)
        terminated = False
        truncated = False

        # --- 1. APLICĂ ACȚIUNEA AGENTULUI ---
        # Schimbare direcție
        if action == 1: # Stânga
            self.agent_heading -= self.turn_rate
        elif action == 2: # Dreapta
            self.agent_heading += self.turn_rate
        
        # Schimbare altitudine (Clamped intre 0 si 2)
        if action == 3: # Urcă
            self.agent_alt = min(2, self.agent_alt + 1)
        elif action == 4: # Coboară
            self.agent_alt = max(0, self.agent_alt - 1)
            
        # Actualizare poziție (Cinematică simplă)
        self.agent_pos[0] += math.cos(self.agent_heading) * self.speed
        self.agent_pos[1] += math.sin(self.agent_heading) * self.speed

        # --- 2. ACTUALIZARE INTRUSI ---
        for intr in self.intruders:
            intr['pos'][0] += intr['vx']
            intr['pos'][1] += intr['vy']
            # Dacă ies din hartă, îi resetăm pe partea cealaltă (wrap around) pt a menține traficul
            if intr['pos'][0] > 100: intr['pos'][0] = 0
            if intr['pos'][0] < 0: intr['pos'][0] = 100
            if intr['pos'][1] > 100: intr['pos'][1] = 0
            if intr['pos'][1] < 0: intr['pos'][1] = 100

        # --- 3. VERIFICĂRI (Coliziuni și Target) ---
        
        # A. Ieșire din hartă
        if not (0 <= self.agent_pos[0] <= 100 and 0 <= self.agent_pos[1] <= 100):
            reward = -1000
            terminated = True
        
        # B. Ajuns la Destinație (rază de 5 unități)
        dist_target = np.linalg.norm(self.agent_pos - self.target_pos)
        if dist_target < 5.0:
            reward = 100
            terminated = True

        # C. Coliziuni cu Intrusii (Logica 2.5D)
        for intr in self.intruders:
            dist_xy = np.linalg.norm(self.agent_pos - intr['pos'])
            dist_alt = abs(self.agent_alt - intr['alt'])
            
            # COLIZIUNE: Prea aproape orizontal ȘI aceeași altitudine
            if dist_xy < self.min_separation and dist_alt == 0:
                reward = -1000
                terminated = True
                break # Nu mai verificăm alții, e gata

        # Limitare pași (să nu se învârtă la infinit)
        if self.step_count > 1000:
            truncated = True

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((600, 600))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 16, bold=True)

        canvas = pygame.Surface((600, 600))
        canvas.fill((20, 20, 30)) # Dark Blue background (Radar screen)
        
        scale = 600 / self.area_size # Convertim metri în pixeli

        # Helper pentru desenare
        def draw_plane(pos, heading, alt, color, is_agent=False):
            cx, cy = int(pos[0] * scale), int(600 - pos[1] * scale) # Inversăm Y pentru ecran
            
            # Desenăm corpul avionului (cerc sau triunghi)
            pygame.draw.circle(canvas, color, (cx, cy), 8)
            
            # Linia de direcție
            end_x = cx + math.cos(heading) * 20
            end_y = cy - math.sin(heading) * 20 # Minus pt că Y e inversat
            pygame.draw.line(canvas, color, (cx, cy), (end_x, end_y), 2)
            
            # --- PARTEA IMPORTANTĂ: SCRIEM ALTITUDINEA ---
            text = self.font.render(f"FL{alt}", True, (255, 255, 255))
            canvas.blit(text, (cx + 10, cy - 10))

        # 1. Desenăm Target
        tx, ty = int(self.target_pos[0] * scale), int(600 - self.target_pos[1] * scale)
        pygame.draw.circle(canvas, (0, 255, 0), (tx, ty), 10, width=2) # Cerc gol verde

        # 2. Desenăm Intrusii (Roșu)
        for intr in self.intruders:
            # Heading-ul intrusului e calculat din viteze
            heading = math.atan2(intr['vy'], intr['vx'])
            # Culoare: Roșu aprins dacă e la aceeași altitudine cu noi, Roșu închis dacă e diferit
            color = (255, 50, 50) if intr['alt'] == self.agent_alt else (100, 30, 30)
            draw_plane(intr['pos'], heading, intr['alt'], color)

        # 3. Desenăm Agentul (Albastru)
        draw_plane(self.agent_pos, self.agent_heading, self.agent_alt, (50, 150, 255), is_agent=True)

        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()