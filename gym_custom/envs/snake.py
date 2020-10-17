import numpy as np
import random
import logging
import math

import gym
from gym import spaces

logger = logging.getLogger(__name__)

STATE_DESK = 0
STATE_BODY = 1
STATE_HEAD = 2
STATE_FOOD = 3

ACTIONS = [[-1, 0], [1, 0], [0, -1], [0, 1]]  # 上/下/左/右
DIR_UP = 0
DIR_DOWN = 1
DIR_LEFT = 2
DIR_RIGHT = 3


class Snake(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.state = np.zeros((height, width)).astype(int)
        self.body = []
        self.food = []
        self.direction = None

    def draw_body(self, tail=None):
        for i, (y, x) in enumerate(self.body):
            if i == len(self.body) - 1:
                self.state[y, x] = STATE_HEAD
                break
            self.state[y, x] = STATE_BODY
        if tail is not None:
            self.state[tail[0], tail[1]] = STATE_DESK
        self.state[self.food[0], self.food[1]] = STATE_FOOD

    def go(self, action):
        # 对向action不起作用
        if action + self.direction != 1 and action + self.direction != 5:
            self.direction = action

        head = self.body[-1]
        next_head = [head[0] + ACTIONS[self.direction][0], head[1] + ACTIONS[self.direction][1]]

        # hit wall
        if next_head[0] < 0 or next_head[0] >= self.height or next_head[1] < 0 or next_head[1] >= self.width:
            return self.state, -1, True, 'Failed. Hit wall'

        # eat itself
        if next_head in self.body[1:]:
            return self.state, -1, True, 'Failed. Eat itself'

        # eat food
        self.body.append(next_head)
        if next_head == self.food:
            self.generate_food()
            self.draw_body()
            return self.state, 1, False, 'Succeed. Eat food'

        # nothing happened
        tail = self.body.pop(0)
        self.draw_body(tail)
        return self.state, self.get_reward(), False, None

    def get_reward(self):
        head = self.body[-1]
        food = self.food
        dis = math.sqrt(pow((food[0] - head[0]), 2) + pow((food[1] - head[1]), 2))  # >= 1
        reward = (1 / dis) * 0.5                                                    # <= 0.5
        return reward

    def generate_snake(self):
        x = random.randint(1, self.width - 2)
        y = random.randint(1, self.height - 2)
        head = [y, x]
        self.direction = random.randint(0, len(ACTIONS) - 1)
        tail = [head[0] - ACTIONS[self.direction][0], head[1] - ACTIONS[self.direction][1]]

        self.body.clear()
        self.body.append(tail)
        self.body.append(head)
        self.state = np.zeros((self.height, self.width)).astype(int)
        self.state[tail[0], tail[1]] = STATE_BODY
        self.state[head[0], head[1]] = STATE_HEAD

    def generate_food(self):
        y, x = self.body[-1]
        while [y, x] in self.body:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
        self.food = [y, x]
        self.state[y, x] = STATE_FOOD

    import numpy as np
    import random
    import logging
    import math

    import gym
    from gym import spaces

    logger = logging.getLogger(__name__)

    STATE_DESK = 0
    STATE_BODY = 1
    STATE_HEAD = 2
    STATE_FOOD = 3

    ACTIONS = [[-1, 0], [1, 0], [0, -1], [0, 1]]  # 上/下/左/右
    DIR_UP = 0
    DIR_DOWN = 1
    DIR_LEFT = 2
    DIR_RIGHT = 3

    class Snake(object):
        def __init__(self, width, height):
            self.width = width
            self.height = height

            self.state = np.zeros((height, width)).astype(int)
            self.body = []
            self.food = []
            self.direction = None

        def draw_body(self, tail=None):
            for i, (y, x) in enumerate(self.body):
                if i == len(self.body) - 1:
                    self.state[y, x] = STATE_HEAD
                    break
                self.state[y, x] = STATE_BODY
            if tail is not None:
                self.state[tail[0], tail[1]] = STATE_DESK
            self.state[self.food[0], self.food[1]] = STATE_FOOD

        def go(self, action):
            # 对向action不起作用
            if action + self.direction != 1 and action + self.direction != 5:
                self.direction = action

            head = self.body[-1]
            next_head = [head[0] + ACTIONS[self.direction][0], head[1] + ACTIONS[self.direction][1]]

            # hit wall
            if next_head[0] < 0 or next_head[0] >= self.height or next_head[1] < 0 or next_head[1] >= self.width:
                return self.state, -1, True, 'Failed. Hit wall'

            # eat itself
            if next_head in self.body[1:]:
                return self.state, -1, True, 'Failed. Eat itself'

            # eat food
            self.body.append(next_head)
            if next_head == self.food:
                self.generate_food()
                self.draw_body()
                return self.state, 1, False, 'Succeed. Eat food'

            # nothing happened
            tail = self.body.pop(0)
            self.draw_body(tail)
            return self.state, self.get_reward(), False, None

        def get_reward(self):
            head = self.body[-1]
            food = self.food
            dis = math.sqrt(pow((food[0] - head[0]), 2) + pow((food[1] - head[1]), 2))  # >= 1
            reward = (1 / dis) * 0.5  # <= 0.5
            return reward

        def generate_snake(self):
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)
            head = [y, x]
            self.direction = random.randint(0, len(ACTIONS) - 1)
            tail = [head[0] - ACTIONS[self.direction][0], head[1] - ACTIONS[self.direction][1]]

            self.body.clear()
            self.body.append(tail)
            self.body.append(head)
            self.state = np.zeros((self.height, self.width)).astype(int)
            self.state[tail[0], tail[1]] = STATE_BODY
            self.state[head[0], head[1]] = STATE_HEAD

        def generate_food(self):
            y, x = self.body[-1]
            while [y, x] in self.body:
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
            self.food = [y, x]
            self.state[y, x] = STATE_FOOD

        def reset(self):
            self.generate_snake()
            self.generate_food()
            self.draw_body()
            return self.state

    class SnakeEnv(gym.Env):
        metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 1
        }

        def __init__(self):

            self.width = 10
            self.height = 10

            self.viewer = None

            self.snake = Snake(self.width, self.height)
            # self.state = self.snake.state
            self.state = self.render(mode='rgb_array')
            self.action_space = spaces.Discrete(len(ACTIONS))

        def _seed(self, seed=None):
            self.np_random, seed = random.seeding.np_random(seed)
            return [seed]

        def step(self, action):
            err_msg = "%r (%s) invalid" % (action, type(action))
            assert self.action_space.contains(action), err_msg

            self.state, reward, done, info = self.snake.go(action)
            # return self.state, reward, done, info
            return self.render(mode='rgb_array'), reward, done, info

        def reset(self):
            self.state = self.snake.reset()
            # return self.state
            return self.render(mode='rgb_array')

        def render(self, mode='human'):
            from gym.envs.classic_control import rendering

            grid = 20
            circle_r = 10

            screen_width = grid * (self.width + 2)
            screen_height = grid * (self.height + 2)

            if self.viewer is None:

                self.viewer = rendering.Viewer(screen_width, screen_height)

                # create mesh world
                lines = []
                # for w in range(1, self.width + 2):
                for w in range(1, self.width + 2, self.width):
                    line = rendering.Line((grid * w, grid), (grid * w, grid * (self.height + 1)))
                    line.set_color(0, 0, 0)
                    lines.append(line)
                    self.viewer.add_geom(line)
                # for h in range(1, self.height + 2):
                for h in range(1, self.height + 2, self.height):
                    line = rendering.Line((grid, grid * h), (grid * (self.width + 1), grid * h))
                    lines[-1].set_color(0, 0, 0)
                    lines.append(line)
                    self.viewer.add_geom(line)

            # create snake body
            for i, (y, x) in enumerate(self.snake.body):
                body = rendering.make_circle(circle_r)
                bodytrans = rendering.Transform(translation=(grid * (1.5 + x), grid * (1.5 + y)))
                body.add_attr(bodytrans)
                body.set_color(0.5, 0.5, 0.5)
                if i == len(self.snake.body) - 1:
                    body.set_color(0, 0, 0)
                self.viewer.add_onetime(body)

            # create food
            if self.snake.food:
                food = rendering.make_circle(circle_r)
                y, x = self.snake.food
                foodtrans = rendering.Transform(translation=(grid * (1.5 + x), grid * (1.5 + y)))
                food.add_attr(foodtrans)
                food.set_color(0, 1, 0)
                self.viewer.add_onetime(food)

            return self.viewer.render(return_rgb_array=mode == 'rgb_array')

        def close(self):
            if self.viewer:
                self.viewer.close()

    def reset(self):
        self.generate_snake()
        self.generate_food()
        self.draw_body()
        return self.state


class SnakeEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1
    }

    def __init__(self):

        self.width = 10
        self.height = 10

        self.viewer = None

        self.snake = Snake(self.width, self.height)
        # self.state = self.snake.state
        self.state = self.render(mode='rgb_array')
        self.action_space = spaces.Discrete(len(ACTIONS))

    def _seed(self, seed=None):
        self.np_random, seed = random.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        self.state, reward, done, info = self.snake.go(action)
        # return self.state, reward, done, info
        return self.render(mode='rgb_array'), reward, done, info

    def reset(self):
        self.state = self.snake.reset()
        # return self.state
        return self.render(mode='rgb_array')

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        grid = 20
        circle_r = 10

        screen_width = grid * (self.width + 2)
        screen_height = grid * (self.height + 2)

        if self.viewer is None:

            self.viewer = rendering.Viewer(screen_width, screen_height)

            # create mesh world
            lines = []
            # for w in range(1, self.width + 2):
            for w in range(1, self.width + 2, self.width):
                line = rendering.Line((grid*w, grid), (grid*w, grid*(self.height + 1)))
                line.set_color(0, 0, 0)
                lines.append(line)
                self.viewer.add_geom(line)
            # for h in range(1, self.height + 2):
            for h in range(1, self.height + 2, self.height):
                line = rendering.Line((grid, grid*h), (grid*(self.width + 1), grid*h))
                lines[-1].set_color(0, 0, 0)
                lines.append(line)
                self.viewer.add_geom(line)

        # create snake body
        for i, (y, x) in enumerate(self.snake.body):
            body = rendering.make_circle(circle_r)
            bodytrans = rendering.Transform(translation=(grid * (1.5 + x), grid * (1.5 + y)))
            body.add_attr(bodytrans)
            body.set_color(0.5, 0.5, 0.5)
            if i == len(self.snake.body) - 1:
                body.set_color(0, 0, 0)
            self.viewer.add_onetime(body)

        # create food
        if self.snake.food:
            food = rendering.make_circle(circle_r)
            y, x = self.snake.food
            foodtrans = rendering.Transform(translation=(grid * (1.5 + x), grid * (1.5 + y)))
            food.add_attr(foodtrans)
            food.set_color(0, 1, 0)
            self.viewer.add_onetime(food)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
