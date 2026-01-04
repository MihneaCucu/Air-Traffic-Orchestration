import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class ATC2DEnv(gym.Env):
    """ATC environment with queue, two departure runways, and priority arrivals"""

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        queue_length=12,
        lane_height=10,
        max_steps=500,
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

        # Actions:
        # 0 = wait
        # 1 = release to lane 0
        # 2 = release to lane 1
        self.action_space = spaces.Discrete(1 + self.num_lanes)

        # Observation:
        # [queue,
        #  dep0, dep0_y,
        #  dep1, dep1_y,
        #  arrival_active, arrival_lane, arrival_y]
        low = np.zeros(9, dtype=np.float32)
        high = np.array(
            [
                queue_length,          # blue queue
                1, lane_height,
                1, lane_height,
                1, self.num_lanes - 1, lane_height,
                queue_length           # landed red planes
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high)

        # Rendering
        self.window = None
        self.clock = None
        self.font = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.planes_in_queue = self.queue_length
        self.arrivals_landed = 0

        # Departures
        self.dep_occupied = [False] * self.num_lanes
        self.dep_y = [0] * self.num_lanes

        # Arrival
        self.arrival_active = False
        self.arrival_lane = 0
        self.arrival_y = self.lane_height

        self.steps = 0

        return self._get_obs(), {}

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
        self.steps += 1
        reward = -0.1
        terminated = False
        truncated = False

        # -------- Agent decision --------
        if action > 0 and not self.arrival_active:
            lane = action - 1

            if self.planes_in_queue > 0 and not self.dep_occupied[lane]:
                # 25% chance an arrival appears instead of takeoff
                if np.random.rand() < 0.25:
                    # Spawn arrival
                    self.arrival_active = True
                    self.arrival_lane = lane
                    self.arrival_y = self.lane_height

                    reward -= 1.0  # penalty for blocking takeoff
                else:
                    # Normal takeoff
                    self.planes_in_queue -= 1
                    self.dep_occupied[lane] = True
                    self.dep_y[lane] = 0

        # -------- Arrival movement (down) --------
        if self.arrival_active:
            self.arrival_y -= 1
            reward -= 0.2  # pressure to clear arrival quickly

            if self.arrival_y <= 0:
                self.arrival_active = False
                self.arrivals_landed += 1
                reward += 5.0  # successful landing

        # -------- Departure movement (up) --------
        for i in range(self.num_lanes):
            # Block only the lane used by the arrival
            if self.arrival_active and i == self.arrival_lane:
                continue

            if self.dep_occupied[i]:
                self.dep_y[i] += 1

                if self.dep_y[i] >= self.lane_height:
                    self.dep_occupied[i] = False
                    self.dep_y[i] = 0
                    reward += 10.0

        # -------- Termination --------
        if (
            self.planes_in_queue == 0
            and not any(self.dep_occupied)
            and not self.arrival_active
        ):
            terminated = True
            reward += 50.0

        if self.steps >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    
    def render(self):
        if self.render_mode != "human":
            return

        width, height = 480, 560
        queue_slot = 28
        slot_margin = 5
        bottom_margin = 80
        top_margin = 60
        runway_width = 50
        runway_gap = 40
        top_queue_slot = 28
        top_queue_margin = 5
        top_queue_y = 20


        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((width, height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 16, bold=True)

        surf = pygame.Surface((width, height))
        surf.fill((15, 18, 28))

        # ---- Queue ----
        start_x = (width - (self.queue_length * queue_slot +
                   (self.queue_length - 1) * slot_margin)) // 2
        y_slot = height - bottom_margin

        for i in range(self.queue_length):
            x = start_x + i * (queue_slot + slot_margin)
            filled = i < self.planes_in_queue
            color = (70, 150, 255) if filled else (60, 65, 80)
            pygame.draw.rect(surf, color, (x, y_slot, queue_slot, queue_slot))
            pygame.draw.rect(surf, (40, 45, 55), (x, y_slot, queue_slot, queue_slot), 1)

        # ---- Arrival queue (top) ----
        max_slots = self.queue_length
        start_x_top = (
            width - (max_slots * top_queue_slot + (max_slots - 1) * top_queue_margin)
        ) // 2

        for i in range(max_slots):
            x = start_x_top + i * (top_queue_slot + top_queue_margin)
            filled = i < self.arrivals_landed
            color = (220, 70, 70) if filled else (80, 50, 50)
            pygame.draw.rect(
                surf,
                color,
                (x, top_queue_y, top_queue_slot, top_queue_slot),
            )
            pygame.draw.rect(
                surf,
                (120, 90, 90),
                (x, top_queue_y, top_queue_slot, top_queue_slot),
                1,
            )

        # ---- Runways ----
        total_width = self.num_lanes * runway_width + (self.num_lanes - 1) * runway_gap
        base_x = width // 2 - total_width // 2
        ry = top_margin
        rh = height - top_margin - bottom_margin - queue_slot

        for i in range(self.num_lanes):
            rx = base_x + i * (runway_width + runway_gap)
            pygame.draw.rect(surf, (55, 65, 85), (rx, ry, runway_width, rh), 2)

            if self.dep_occupied[i]:
                p = self.dep_y[i] / max(1, self.lane_height)
                py = ry + rh - 36 - int(p * (rh - 36))
                pygame.draw.rect(surf, (230, 180, 80), (rx + 7, py, runway_width - 14, 30))

        # ---- Arrival ----
        if self.arrival_active:
            rx = base_x + self.arrival_lane * (runway_width + runway_gap)
            p = self.arrival_y / max(1, self.lane_height)
            py = ry + int(p * (rh - 36))
            pygame.draw.rect(surf, (220, 70, 70), (rx + 7, py, runway_width - 14, 30))

        # ---- Info ----
        info = self.font.render(
            f"Queue: {self.planes_in_queue} | Arriving: {self.arrival_active}",
            True,
            (220, 220, 230),
        )
        surf.blit(info, (10, 10))

        self.window.blit(surf, (0, 0))
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None
