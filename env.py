import numpy as np
import gym
from gym.spaces import Dict, Box, Discrete


class QEnv(gym.Env):

    GRID_SIZE = 10

    def __init__(self, full_info=False):
        super().__init__()

        self.num_obstacles = 4
        self.used_obstacles = 2
        self.obstance_density = .5
        self.mods = [
            QEnv.mod_bounce,
            QEnv.mod_null,
            QEnv.mod_penalty,
            QEnv.mod_slide,
            QEnv.mod_teleport,
            QEnv.mod_terminate
        ]

        self.observation_space = Dict({
            "obs": Box(0, 1, shape=(QEnv.GRID_SIZE, QEnv.GRID_SIZE, self.num_obstacles+2), dtype=np.int64),
            "info": Box(0, 1, shape=(self.num_obstacles, len(self.mods), 2), dtype=np.int64)
        })
        self.action_space = Discrete(4 + self.num_obstacles if full_info else 4)

        self.full_info = full_info

        self.grid = np.zeros((QEnv.GRID_SIZE, QEnv.GRID_SIZE, self.num_obstacles+2))
        self.info = np.zeros((self.num_obstacles, len(self.mods), 2))
        if self.full_info:
            self.info_mask = np.ones((self.num_obstacles, len(self.mods)))
        else:
            self.info_mask = np.zeros((self.num_obstacles, len(self.mods)))

    def reset(self):
        self.grid = np.zeros((QEnv.GRID_SIZE, QEnv.GRID_SIZE, self.num_obstacles+2))
        start = tuple(np.random.randint(QEnv.GRID_SIZE, size=2))
        goal = tuple(np.random.randint(QEnv.GRID_SIZE, size=2))
        while np.array_equal(start, goal):
            goal = tuple(np.random.randint(QEnv.GRID_SIZE, size=2))
        self.grid[start+(0,)] = 1
        self.grid[goal+(1,)] = 1

        obstancles = np.random.choice(
            list(range(self.num_obstacles)), 
            size=self.used_obstacles, 
            replace=False
        )

        for x in range(QEnv.GRID_SIZE):
            for y in range(QEnv.GRID_SIZE):

                if np.random.rand() < self.obstance_density and \
                    self.grid[x, y, 0] == 0 and \
                    self.grid[x, y, 1] == 0:
                    
                    obst = np.random.choice(obstancles)
                    self.grid[x, y, obst+2] = 1

        self.info = np.zeros((self.num_obstacles, len(self.mods), 2))
        for obst in range(self.num_obstacles):
            mod = np.random.randint(len(self.mods))
            self.info[obst, :, 0] = 1
            self.info[obst, mod, 0] = 0
            self.info[obst, mod, 1] = 1

        if self.full_info:
            self.info_mask = np.ones((self.num_obstacles, len(self.mods)))
        else:
            self.info_mask = np.zeros((self.num_obstacles, len(self.mods)))

        return self.get_full_obs()

    def step(self, action):

        if action < 4:

            y, x = QEnv.get_transition(self.grid, action)

            self.grid[:, :, 0] = 0
            self.grid[y, x, 0] = 1

            reward = 0
            done = False

            obst = np.where(self.grid[y, x, 2:])[0]
            while obst.size > 0 and not done:
                mod_vector = np.where(self.info[obst.item()])[1]
                mod_idx = np.where(mod_vector)[0]
                mod_fun = self.mods[mod_idx.item()]
                g, r, done = mod_fun(np.copy(self.grid), action)

                reward += r

                if np.array_equal(g, self.grid):
                    break
                self.grid = g
                y, x = self.current_pos()

                obst = np.where(self.grid[y, x, 2:])[0]

            if np.where(self.grid[:, :, 0]) == np.where(self.grid[:, :, 1]):
                done = True
                reward += 1

        else:

            self.info_mask[action-4, :] = 1
            reward = 0
            done = False

        return self.get_full_obs(), reward, done, dict()

    def render(self):
        print(self.grid)
        symbols = ["@", "G"] + [str(i) for i in range(self.num_obstacles)]

        out = ""
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                cell = self.grid[y, x]
                if np.any(cell):
                    out += symbols[np.where(cell)[0][0]]
                else:
                    out += "."
            out += "\n"

        print(out)

    def get_full_obs(self):
        masked_info = self.info * np.expand_dims(self.info_mask, -1)
        full_obs = {
            "obs": self.grid,
            "info": masked_info
        }
        return full_obs

    def current_pos(self):
        return self.get_pos(self.grid)

    @staticmethod
    def get_transition(grid, action):
        y, x = QEnv.get_pos(grid)
        if action == 0:
            y -= 1
        elif action == 1:
            x += 1
        elif action == 2:
            y += 1
        elif action == 3:
            x -=1
        y = np.clip(y, 0, QEnv.GRID_SIZE-1)
        x = np.clip(x, 0, QEnv.GRID_SIZE-1)
        return y, x

    @staticmethod
    def get_pos(grid):
        indices = np.where(grid[:, :, 0])
        return indices[0].item(), indices[1].item()
        
    @staticmethod
    def mod_bounce(grid, action):
        reverse = [2, 3, 0, 1]
        y, x = QEnv.get_transition(grid, reverse[action])
        grid[:, :, 0] = 0
        grid[y, x, 0] = 1
        return grid, 0, False
        
    @staticmethod
    def mod_null(grid, action):
        return grid, 0, False
        
    @staticmethod
    def mod_penalty(grid, action):
        return grid, -1, False
        
    @staticmethod
    def mod_slide(grid, action):
        y, x = QEnv.get_transition(grid, action)
        grid[:, :, 0] = 0
        grid[y, x, 0] = 1
        return grid, 0, False

    @staticmethod
    def mod_teleport(grid, action):
        y, x = tuple(np.random.randint(QEnv.GRID_SIZE, size=2))
        grid[:, :, 0] = 0
        grid[y, x, 0] = 1
        return grid, 0, False
        
    @staticmethod
    def mod_terminate(grid, action):
        return grid, -1, True
