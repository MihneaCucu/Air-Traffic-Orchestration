import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import time
import math

class ATC2DEnv(gym.Env):
    """
    Enhanced ATC Environment with Improved UI.
    Features: 
    - Enhanced Weather (Rain/Wind/Clouds) & Safety Violations
    - 3D Altitude Effects with realistic shadows
    - Realistic Takeoff/Landing Physics (Rolling vs Flying)
    - Beautiful UI with information panels
    - Smooth animations and visual feedback
    - Sound-ready architecture
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        queue_length=12,
        lane_height=10,
        max_steps=1000,
        arrival_prob=0.02,
        render_mode=None,
    ):
        super().__init__()

        self.queue_length = queue_length
        self.lane_height = lane_height
        self.max_steps = max_steps
        self.arrival_prob = arrival_prob
        self.render_mode = render_mode

        self.num_lanes = 2

        # Actions: 0=wait, 1=lane0, 2=lane1
        self.action_space = spaces.Discrete(1 + self.num_lanes)

        # Observation space 
        low = np.zeros(9, dtype=np.float32)
        high = np.array(
            [
                queue_length,          
                1, lane_height,
                1, lane_height,
                1, self.num_lanes - 1, lane_height,
                queue_length           
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high)

        # Rendering
        self.window = None
        self.clock = None
        self.font = None
        self.title_font = None
        self.small_font = None
        
        # Internal Complexity State
        self.anim_frames = 8  # Smoother animations
        self.wind_speed = 0.0 
        self.rain_drops = []
        self.clouds = []
        self.near_misses = 0
        self.fuel_timers = []
        self.last_action_text = "WAITING"
        self.action_timer = 0
        self.alert_flash = 0
        self.total_departures = 0
        self.safety_score = 100 
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.planes_in_queue = self.queue_length
        self.arrivals_landed = 0
        self.total_departures = 0
        self.dep_occupied = [False] * self.num_lanes
        self.dep_y = [0.0] * self.num_lanes
        
        self.arrival_active = False
        self.arrival_lane = 0
        self.arrival_y = float(self.lane_height)
        
        # Init Fuel
        self.fuel_timers = [np.random.randint(20, 100) for _ in range(self.queue_length)]

        # Previous states for interpolation
        self.prev_dep_y = [0.0] * self.num_lanes
        self.prev_arrival_y = float(self.lane_height)
        self.prev_arrival_active = False
        self.prev_dep_occupied = [False] * self.num_lanes

        self.steps = 0
        self.near_misses = 0
        self.safety_score = 100
        self.wind_speed = np.random.uniform(0.0, 0.4)
        self.last_action_text = "SYSTEM INITIALIZED"
        self.action_timer = 30
        self.alert_flash = 0
        self._init_rain()
        self._init_clouds()
        
        return self._get_obs(), {}

    def _init_rain(self):
        self.rain_drops = []
        for _ in range(80):
            self.rain_drops.append([random.randint(0, 600), random.randint(0, 700)])
    
    def _init_clouds(self):
        """Initialize cloud positions for atmospheric effect"""
        self.clouds = []
        for _ in range(5):
            self.clouds.append({
                'x': random.randint(-50, 650),
                'y': random.randint(50, 250),
                'size': random.randint(40, 80),
                'speed': random.uniform(0.1, 0.3)
            })

    def _get_obs(self):
        return np.array(
            [
                self.planes_in_queue,
                int(self.dep_occupied[0]),
                self.dep_y[0] if self.dep_occupied[0] else 0,
                int(self.dep_occupied[1]),
                self.dep_y[1] if self.dep_occupied[1] else 0,
                int(self.arrival_active),
                self.arrival_lane if self.arrival_active else 0,
                self.arrival_y if self.arrival_active else 0,
                self.arrivals_landed,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        # Store previous state
        self.prev_dep_y = list(self.dep_y)
        self.prev_dep_occupied = list(self.dep_occupied)
        self.prev_arrival_y = self.arrival_y
        self.prev_arrival_active = self.arrival_active

        self.steps += 1
        reward = -0.01  # Reduced time penalty
        terminated = False
        truncated = False

        # 1. Update Weather 
        self.wind_speed += np.random.uniform(-0.05, 0.05)
        self.wind_speed = np.clip(self.wind_speed, 0.0, 0.6)

        # 2. Update Fuel (reduced penalty)
        for i in range(self.planes_in_queue):
            self.fuel_timers[i] -= 1
            if self.fuel_timers[i] < 5:
                reward -= 0.2  # Reduced from 0.5 

        # -------- Agent decision --------
        if action > 0 and not self.arrival_active:
            lane = action - 1
            if self.planes_in_queue > 0 and not self.dep_occupied[lane]:
                # Spawn Arrival Logic (reduced randomness)
                if np.random.rand() < 0.20:  # Reduced from 0.25
                    self.arrival_active = True
                    self.arrival_lane = lane
                    self.arrival_y = float(self.lane_height)
                    reward -= 0.2  # Reduced penalty
                    self.last_action_text = f"ARRIVAL on RUNWAY {lane + 1}"
                    self.action_timer = 30
                    self.alert_flash = 15
                else:
                    # Takeoff
                    self.planes_in_queue -= 1
                    self.total_departures += 1
                    self.fuel_timers.pop(0)
                    self.fuel_timers.append(np.random.randint(20, 100))
                    
                    self.dep_occupied[lane] = True
                    self.dep_y[lane] = 0.0
                    reward += 1.0  # Small immediate reward for action
                    self.last_action_text = f"DEPARTURE cleared RUNWAY {lane + 1}"
                    self.action_timer = 30
        elif action == 0:
            self.last_action_text = "HOLDING"
            self.action_timer = max(15, self.action_timer)

        # -------- Arrival movement --------
        if self.arrival_active:
            move_speed = 1.0 - (self.wind_speed * 0.5) 
            self.arrival_y -= move_speed
            reward -= 0.02  # Reduced from 0.1
            
            if self.arrival_y <= 0:
                self.arrival_active = False
                self.arrivals_landed += 1
                reward += 15.0  # Increased reward for successful landing
                self.last_action_text = "LANDING SUCCESSFUL!"
                self.action_timer = 30

        # -------- Departure movement --------
        for i in range(self.num_lanes):
            if self.arrival_active and i == self.arrival_lane:
                continue
            if self.dep_occupied[i]:
                move_speed = 1.0 - (np.random.uniform(0.0, 0.2)) 
                self.dep_y[i] += move_speed
                
                if self.dep_y[i] >= self.lane_height:
                    self.dep_occupied[i] = False
                    self.dep_y[i] = 0.0
                    reward += 15.0  # Increased reward to match landings
                    self.last_action_text = f"DEPARTURE cleared RUNWAY {i + 1}"
                    self.action_timer = 30

        # -------- PROXIMITY VIOLATIONS --------
        if all(self.dep_occupied):
            if abs(self.dep_y[0] - self.dep_y[1]) < 2.0:
                reward -= 5.0  # Increased penalty for violations
                self.near_misses += 1
                self.safety_score = max(0, self.safety_score - 5)
                self.last_action_text = "⚠ PROXIMITY ALERT!"
                self.action_timer = 30
                self.alert_flash = 20
        
        if self.arrival_active:
            other_lane = 1 - self.arrival_lane
            if self.dep_occupied[other_lane]:
                if abs(self.arrival_y - self.dep_y[other_lane]) < 2.0:
                     reward -= 10.0  # Increased penalty for critical violations
                     self.near_misses += 1
                     self.safety_score = max(0, self.safety_score - 10)
                     self.last_action_text = "⚠⚠ CRITICAL VIOLATION!"
                     self.action_timer = 30
                     self.alert_flash = 30

        # -------- Termination --------
        if (self.planes_in_queue == 0 and not any(self.dep_occupied) and not self.arrival_active):
            terminated = True
            reward += 100.0  # Increased completion bonus

        if self.steps >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    # =========================================================================
    # ENHANCED RENDER LOGIC WITH BEAUTIFUL UI
    # =========================================================================
    
    def _draw_jet(self, surface, x, y, color, heading="up", scale=1.0, fuel_critical=False, altitude=0.0):
        """
        Enhanced jet drawing with altitude effects.
        altitude: 0.0 (Ground) to 1.0 (High).
        Effect: Shifts shadow, scales plane, and adds glow.
        """
        # 1. Weather Shake
        if self.wind_speed > 0.3 and altitude > 0.1:
            x += random.randint(-2, 2)
            y += random.randint(-1, 1)

        # 2. Altitude Effects
        visual_scale = scale * (1.0 - (0.12 * altitude)) 
        shadow_offset = 6 + (25 * altitude)
        
        # Enhanced Plane Shape
        body_points = [
            (0, -20), (5, -10), (18, 5), (6, 6), (6, 14), (12, 19), 
            (0, 16), (-12, 19), (-6, 14), (-6, 6), (-18, 5), (-5, -10)
        ]
        
        # Critical Fuel Flash
        final_color = color
        if fuel_critical:
            flash = (time.time() * 8) % 2 > 1
            final_color = (255, 255, 100) if flash else (255, 30, 30)

        # Transform Points
        plane_poly = []
        shadow_poly = []
        
        for px, py in body_points:
            px *= visual_scale
            py *= visual_scale
            if heading == "down": 
                py = -py
            
            plane_poly.append((x + px, y + py))
            shadow_poly.append((x + px + shadow_offset, y + py + shadow_offset))

        # Draw soft shadow with gradient effect
        shadow_alpha = int(80 * (1 - altitude * 0.5))
        shadow_surf = pygame.Surface((100, 100), pygame.SRCALPHA)
        pygame.draw.polygon(shadow_surf, (*color[:3], shadow_alpha), 
                          [(p[0] - x + 50, p[1] - y + 50) for p in shadow_poly])
        surface.blit(shadow_surf, (x - 50, y - 50))
        
        # Draw Plane with outline
        pygame.draw.polygon(surface, final_color, plane_poly)
        pygame.draw.polygon(surface, (255, 255, 255), plane_poly, 2)
        
        # Add cockpit window
        if heading == "up":
            pygame.draw.circle(surface, (100, 200, 255), (int(x), int(y - 12 * visual_scale)), 
                             int(3 * visual_scale))
        else:
            pygame.draw.circle(surface, (100, 200, 255), (int(x), int(y + 12 * visual_scale)), 
                             int(3 * visual_scale))
    
    def _draw_cloud(self, surface, x, y, size):
        """Draw a fluffy cloud"""
        alpha = 120
        cloud_surf = pygame.Surface((size * 2, size), pygame.SRCALPHA)
        # Multiple overlapping circles for cloud effect
        positions = [(0, size//2), (size//3, size//3), (size*2//3, size//3), 
                    (size, size//2), (size//2, size//4)]
        for cx, cy in positions:
            pygame.draw.circle(cloud_surf, (200, 210, 220, alpha), (cx, cy), size//3)
        surface.blit(cloud_surf, (x - size, y - size//2))
    
    def _draw_info_panel(self, surface, x, y, w, h, title, lines, color_scheme="blue"):
        """Draw a modern information panel"""
        # Background with border
        bg_color = (25, 30, 40, 200)
        border_colors = {
            "blue": (50, 150, 255),
            "green": (50, 255, 150),
            "red": (255, 80, 80),
            "yellow": (255, 200, 50)
        }
        border_color = border_colors.get(color_scheme, (50, 150, 255))
        
        # Panel background
        panel_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(panel_surf, bg_color, (0, 0, w, h), border_radius=8)
        pygame.draw.rect(panel_surf, border_color, (0, 0, w, h), 2, border_radius=8)
        
        # Title bar
        pygame.draw.rect(panel_surf, (*border_color, 100), (0, 0, w, 25), border_radius=8)
        pygame.draw.line(panel_surf, border_color, (0, 25), (w, 25), 2)
        
        surface.blit(panel_surf, (x, y))
        
        # Title text
        title_txt = self.small_font.render(title, True, (255, 255, 255))
        surface.blit(title_txt, (x + 10, y + 5))
        
        # Content lines
        for i, line in enumerate(lines):
            txt = self.small_font.render(line, True, (220, 230, 240))
            surface.blit(txt, (x + 10, y + 35 + i * 20))

    def render(self):
        if self.render_mode != "human":
            return

        width, height = 800, 700
        if self.window is None:
            pygame.init()
            pygame.display.set_caption("✈ ATC Control Tower - Enhanced Edition")
            self.window = pygame.display.set_mode((width, height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 14)
            self.small_font = pygame.font.SysFont("Arial", 12)
            self.title_font = pygame.font.SysFont("Arial", 22, bold=True)
            self.large_font = pygame.font.SysFont("Arial", 18, bold=True)

        # Decrease timers
        if self.action_timer > 0:
            self.action_timer -= 1
        if self.alert_flash > 0:
            self.alert_flash -= 1

        # Animation Loop with smoother frames
        frames = self.anim_frames 
        for f in range(1, frames + 1):
            
            # Event handling to prevent "not responding"
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    exit()

            progress = f / frames 

            # === BACKGROUND ===
            # Dynamic gradient background based on weather
            surf = pygame.Surface((width, height))
            if self.wind_speed > 0.4:
                # Storm colors
                for y in range(height):
                    ratio = y / height
                    r = int(15 + ratio * 10)
                    g = int(20 + ratio * 15)
                    b = int(25 + ratio * 20)
                    pygame.draw.line(surf, (r, g, b), (0, y), (width, y))
            else:
                # Clear sky gradient
                for y in range(height):
                    ratio = y / height
                    r = int(30 + ratio * 20)
                    g = int(40 + ratio * 30)
                    b = int(60 + ratio * 40)
                    pygame.draw.line(surf, (r, g, b), (0, y), (width, y))

            # === CLOUDS ===
            for cloud in self.clouds:
                self._draw_cloud(surf, int(cloud['x']), int(cloud['y']), cloud['size'])
                cloud['x'] += cloud['speed'] + self.wind_speed * 0.3
                if cloud['x'] > width + 100:
                    cloud['x'] = -100
                    cloud['y'] = random.randint(50, 250)

            # === RAIN ===
            if self.wind_speed > 0.15:
                for i in range(len(self.rain_drops)):
                    rx, ry = self.rain_drops[i]
                    ry += 25 + int(self.wind_speed * 10)
                    rx += int(self.wind_speed * 20)
                    if ry > height: 
                        ry = random.randint(-30, 0)
                    if rx > width: 
                        rx = random.randint(0, width)
                    if rx < 0: 
                        rx = width
                    self.rain_drops[i] = [rx, ry]
                    
                    # Draw rain with slight transparency
                    rain_color = (100, 120, 140)
                    pygame.draw.line(surf, rain_color, (rx, ry), 
                                   (rx - int(self.wind_speed * 8), ry - 18), 2)

            # === RUNWAY LAYOUT ===
            runway_area_start = 180
            top_margin = 100
            bottom_margin = 120
            runway_w = 100
            runway_gap = 120
            runway_h = height - top_margin - bottom_margin
            total_rw_w = self.num_lanes * runway_w + (self.num_lanes - 1) * runway_gap
            start_x = (width - total_rw_w) // 2 + 50
            
            plane_positions = []

            # Draw runways with enhanced graphics
            for i in range(self.num_lanes):
                rx = start_x + i * (runway_w + runway_gap)
                ry = top_margin
                
                # Runway with gradient
                for stripe_y in range(0, runway_h, 2):
                    shade = 50 + (stripe_y % 20)
                    pygame.draw.line(surf, (shade, shade + 5, shade + 10), 
                                   (rx, ry + stripe_y), (rx + runway_w, ry + stripe_y))
                
                # Runway border
                pygame.draw.rect(surf, (80, 85, 90), (rx, ry, runway_w, runway_h), 3)
                
                # Centerline dashes
                for dy in range(0, runway_h, 50):
                    pygame.draw.line(surf, (200, 200, 50), 
                                   (rx + runway_w//2 - 2, ry + dy), 
                                   (rx + runway_w//2 - 2, ry + dy + 25), 4)
                
                # Runway numbers
                rw_num_txt = self.large_font.render(f"RW{i+1}", True, (255, 255, 100))
                surf.blit(rw_num_txt, (rx + runway_w//2 - 25, ry + runway_h - 40))
                
                # === DEPARTURE PLANES ===
                if self.dep_occupied[i] or (self.prev_dep_occupied[i] and not self.dep_occupied[i]):
                    curr = self.dep_y[i]
                    prev = self.prev_dep_y[i]
                    
                    if not self.dep_occupied[i]: 
                        vis_y = prev + (1.0 * progress)
                    elif not self.prev_dep_occupied[i]: 
                        vis_y = curr * progress
                    else: 
                        vis_y = prev + (curr - prev) * progress

                    # Altitude calculation
                    alt = 0.0
                    if vis_y > 3.0:
                        alt = (vis_y - 3.0) / 7.0
                    alt = min(1.0, max(0.0, alt))

                    pct = vis_y / max(1, self.lane_height)
                    pct = min(1.3, max(0, pct)) 
                    screen_y = (ry + runway_h - 50) - (pct * (runway_h - 100))
                    px = rx + runway_w//2
                    
                    # Draw exhaust trail
                    if alt > 0.3:
                        for trail_i in range(3):
                            trail_y = screen_y + (trail_i + 1) * 15
                            trail_alpha = int(80 - trail_i * 25)
                            trail_surf = pygame.Surface((10, 10), pygame.SRCALPHA)
                            pygame.draw.circle(trail_surf, (200, 200, 200, trail_alpha), (5, 5), 5)
                            surf.blit(trail_surf, (px - 5, trail_y))
                    
                    self._draw_jet(surf, px, screen_y, (255, 180, 0), "up", 1.5, altitude=alt)
                    plane_positions.append((px, screen_y, 'dep', i))

                # === ARRIVAL PLANES ===
                if self.arrival_active or (self.prev_arrival_active and not self.arrival_active):
                    if self.arrival_lane == i or (not self.arrival_active and self.arrival_lane == i):
                        curr = self.arrival_y
                        prev = self.prev_arrival_y
                        
                        if not self.arrival_active: 
                            vis_y = prev - (1.0 * progress)
                        elif not self.prev_arrival_active: 
                            vis_y = self.lane_height 
                        else: 
                            vis_y = prev + (curr - prev) * progress
                            
                        # Altitude for arrivals
                        alt = 0.0
                        if vis_y > 2.0:
                            alt = (vis_y - 2.0) / 8.0
                        alt = min(1.0, max(0.0, alt))

                        pct = vis_y / max(1, self.lane_height)
                        screen_y = (ry + runway_h - 50) - (pct * (runway_h - 100))
                        px = rx + runway_w//2
                        
                        # Approach path indicator
                        if alt > 0.5:
                            pygame.draw.line(surf, (100, 255, 100, 100), 
                                           (px, screen_y), (px, ry + runway_h - 50), 1)
                        
                        self._draw_jet(surf, px, screen_y, (255, 50, 80), "down", 1.5, altitude=alt)
                        plane_positions.append((px, screen_y, 'arr', i))

            # === PROXIMITY ALERT VISUALIZATION ===
            if len(plane_positions) >= 2:
                for i in range(len(plane_positions)):
                    for j in range(i + 1, len(plane_positions)):
                        p1 = plane_positions[i]
                        p2 = plane_positions[j]
                        dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                        
                        if dist < 80:
                            # Draw warning line
                            line_color = (255, 0, 0) if self.alert_flash > 0 else (255, 100, 0)
                            pygame.draw.line(surf, line_color, (p1[0], p1[1]), (p2[0], p2[1]), 3)
                            
                            # Warning text
                            if self.alert_flash > 0 and self.alert_flash % 4 < 2:
                                warn_txt = self.font.render("⚠ PROXIMITY!", True, (255, 255, 0))
                                surf.blit(warn_txt, ((p1[0] + p2[0])//2 - 40, (p1[1] + p2[1])//2 - 10))

            # === LEFT INFO PANEL - LANDED AIRCRAFT ===
            panel_x = 10
            panel_y = 10
            self._draw_info_panel(surf, panel_x, panel_y, 180, 160, 
                                "✈ ARRIVALS", 
                                [f"Landed: {self.arrivals_landed}",
                                 f"Departures: {self.total_departures}",
                                 f"In Queue: {self.planes_in_queue}",
                                 "",
                                 ""],
                                "green")
            
            # Draw landed planes icons below the text with more spacing
            for k in range(min(self.arrivals_landed, 10)):
                px = panel_x + 15 + (k % 5) * 32
                py = panel_y + 110 + (k // 5) * 28
                self._draw_jet(surf, px, py, (50, 255, 150), "down", 0.5)

            # === TOP CENTER - STATUS PANEL ===
            status_w = 350
            status_x = (width - status_w) // 2
            status_y = 10
            
            # Weather info
            wind_pct = int(self.wind_speed * 100)
            if wind_pct < 20:
                weather_color = "green"
                weather_text = "CLEAR"
            elif wind_pct < 40:
                weather_color = "yellow"
                weather_text = "WINDY"
            else:
                weather_color = "red"
                weather_text = "STORM"
            
            # Safety score color
            if self.safety_score > 80:
                safety_color = "green"
            elif self.safety_score > 50:
                safety_color = "yellow"
            else:
                safety_color = "red"
            
            self._draw_info_panel(surf, status_x, status_y, status_w, 95,
                                "⚡ CONTROL CENTER",
                                [f"Weather: {weather_text} ({wind_pct} km/h)",
                                 f"Safety Score: {self.safety_score}/100",
                                 f"Violations: {self.near_misses}",
                                 ""],
                                weather_color)

            # === BOTTOM - DEPARTURE QUEUE ===
            queue_panel_y = height - 110
            self._draw_info_panel(surf, 10, queue_panel_y, width - 20, 100,
                                "🛫 DEPARTURE QUEUE",
                                [],
                                "blue")
            
            queue_y = queue_panel_y + 35
            for k in range(self.queue_length):
                qx = 25 + k * 60
                
                # Queue slot
                slot_color = (40, 45, 50) if k >= self.planes_in_queue else (60, 140, 220)
                pygame.draw.rect(surf, slot_color, (qx, queue_y, 50, 50), 0, border_radius=5)
                pygame.draw.rect(surf, (100, 150, 200), (qx, queue_y, 50, 50), 2, border_radius=5)
                
                if k < self.planes_in_queue:
                    # Fuel bar
                    if k < len(self.fuel_timers):
                        fuel_pct = self.fuel_timers[k] / 100.0
                        bar_width = int(45 * fuel_pct)
                        bar_color = (50, 255, 50) if fuel_pct > 0.3 else (255, 50, 50)
                        pygame.draw.rect(surf, bar_color, (qx + 2, queue_y + 2, bar_width, 4))
                        
                        is_critical = self.fuel_timers[k] < 15
                    else:
                        is_critical = False
                    
                    # Draw plane in queue
                    self._draw_jet(surf, qx + 25, queue_y + 30, (0, 180, 255), "up", 0.6, 
                                 fuel_critical=is_critical)
                    
                    # Position number
                    pos_txt = self.small_font.render(f"{k+1}", True, (200, 200, 200))
                    surf.blit(pos_txt, (qx + 20, queue_y + 48))

            # === ACTION FEEDBACK ===
            if self.action_timer > 0:
                action_alpha = min(255, self.action_timer * 8)
                action_surf = pygame.Surface((300, 40), pygame.SRCALPHA)
                
                bg_alpha = min(200, action_alpha)
                if self.alert_flash > 0:
                    bg_color = (200, 50, 50, bg_alpha)
                else:
                    bg_color = (50, 100, 200, bg_alpha)
                
                pygame.draw.rect(action_surf, bg_color, (0, 0, 300, 40), border_radius=8)
                
                surf.blit(action_surf, (width - 310, status_y + 90))
                
                action_txt = self.font.render(self.last_action_text, True, (255, 255, 255))
                surf.blit(action_txt, (width - 300, status_y + 102))

            # === FINAL BLIT ===
            self.window.blit(surf, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None