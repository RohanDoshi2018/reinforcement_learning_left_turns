from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from operator import add

class LeftTurnEnv(MiniGridEnv):
    """
    Square grid environment where agent ("car") moves from bottom left to bottom right
    of grid, choosing between 1 or 2 squares per step. There are obstacles ("bullets") 
    moving downwards 1 or 2 squares per step, which are stochastically modelled via
    a agression parameter. Bullets with higher aggression aim to collide with the 
    agent.
    """
    def __init__(
            self,
            size=9,
            agent_start_pos=(1, 1), # unused
            agent_start_dir=0, # unused
            n_obstacles=1,
            show_obstacles=True
    ):
        self.agent_start_pos = agent_start_pos # unused
        self.agent_start_dir = agent_start_dir # unused
        self.show_obstacles = show_obstacles

        # Reduce obstacles if there are too many
        if n_obstacles <= size/2 + 1:
            self.n_obstacles = int(n_obstacles)
        else:
            self.n_obstacles = int(size/2)
        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True,
        )
        # Allow only 2 actions permitted: left, right
        self.action_space = spaces.Discrete(self.actions.right + 1)
        self.reward_range = (-1, 1)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-left corner
        self.grid.set(0, height-2, Goal())

        # Place the agent
        self.start_pos = (width-2, height-2)
        self.start_dir = 2
        # TODO(rohandoshi): set the agent into the grid

        # Place the traffic
        if self.show_obstacles:
            aggression = self._rand_float(.8,1)
            self.traffic = Ball(aggression=aggression)

            pos = 1, 1
            self.grid.set(*pos, self.traffic)
            self.traffic.cur_pos = pos


    def step(self, action): # TODO(rohandoshi): add history parameter, add to encoding... 
    # Feed the env an action and get an updated state (encoded) and reward.

        # If invalid action, move forward 1 step by default
        if action >= self.action_space.n:
            action = 1

        reward = 0
        done = False

        # Get current agent position
        c, r = self.agent_pos

        #  Update car position state
        if action == self.actions.left: # Move 2 Forward
            fwd_pos = (max(c-2,0), r)
            fwd_cell = self.grid.get(*fwd_pos)
        elif action == self.actions.right: # Move 1 Forward
            fwd_pos = (c-1, r)
            fwd_cell = self.grid.get(*fwd_pos)
        else:
            assert False, "unknown action"

        if fwd_cell == None or fwd_cell.can_overlap():
            self.agent_pos = fwd_pos
            # TODO(rohandoshi): set the agent into the grid
        if fwd_cell != None and fwd_cell.type == 'goal':
            done = True
            reward = self._reward()

        # Update traffic position state
        if self.show_obstacles:
            self.grid.set(*self.traffic.cur_pos, None)
            self.update_traffic()

            if np.array_equal(self.traffic.cur_pos, self.agent_pos):
                reward = -1
                done = True

        # Encode grid state into an observation
        dc = abs(self.agent_pos[0]-self.traffic.cur_pos[0])
        dr = abs(self.agent_pos[1]-self.traffic.cur_pos[1])
        obs = np.array([dc, dr])

        # Update step counter
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        return obs, reward, done, {}

    def update_traffic(self):
        # Adjust traffic speed based on aggression
        aggression = self.traffic.aggression # [0, 1]

        # We calculate the distance from the point of intersection
        # for both the agent (dc) and traffic (dr). Agressive traffic 
        # drivers will maxamize the liklihood of a collision. Thus, to 
        # increase aggression, we must minimize dc-dw.

        dc = abs(self.agent_pos[0]-self.traffic.cur_pos[0])
        dr = abs(self.agent_pos[1]-self.traffic.cur_pos[1])

        # Assign probabilites for traffic's two choices: slow (1 step) and 
        # fast (2 steps). We calculate this by generating a probability cutoff
        # [0,1] between these two choices. Scale each actions by aggression.


        if dc == 0 and dr == 0:
            cutoff = .5
        else:
            scaling = (dc-dr) / (dc+dr) # [-1, 1]
            cutoff = .5 + aggression * .5 * scaling

        col, row = self.traffic.cur_pos
        rand = self._rand_float(0,1)

        if rand > cutoff:
            row += 2 # fast
        else:
            row += 1 # slow

        if row+1>= self.grid.height:
            row = 1

        # Update the traffic position
        pos = (col, row)
        self.grid.set(*pos, self.traffic)
        self.traffic.cur_pos = pos

    def _reward(self):
        """
        Compute the reward to be given upon success
        """

        return 1 - 0.015 * self.step_count

register(
    id='MiniGrid-LeftTurn-v0',
    entry_point='gym_minigrid.envs:LeftTurnEnv'
)

