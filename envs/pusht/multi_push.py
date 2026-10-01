import collections
import cv2
import gym
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
import shapely.geometry as sg
from gym import spaces
from pymunk.vec2d import Vec2d

from envs.pusht.pymunk_override import DrawOptions

def pymunk_to_shapely(body, shapes):
    geoms = list()
    vertices = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):            
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
            vertices += [verts]
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom, vertices

COLORS = {
    "WHITE":   (255, 255, 255),
    "BLACK":   (0, 0, 0),
    "RED":     (255, 0, 0),
    "GREEN":   (0, 255, 0),
    "BLUE":    (0, 0, 255),
    "YELLOW":  (255, 255, 0),
    "CYAN":    (0, 255, 255),
    "MAGENTA": (255, 0, 255),
    "GRAY":    (128, 128, 128),
    "ORANGE":  (255, 165, 0),
    "PURPLE":  (128, 0, 128),
    "BROWN":   (165, 42, 42)
}


class MultiPushEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,            
            legacy=False, 
            block_cog=None, damping=None,
            render_action=True,
            render_size=96,
            reset_to_state=None,
            signal_idx=None):
        self._seed = None
        self.seed()
        self.window_size = ws = 520   # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        self.red_done = False
        self.goal_poses = [(50,200), (250,200), (450, 200)]
        self.signal_circle_poses = [(50,80), (250,80), (450, 80)]
        self.signal_circle_radius = 20
        if signal_idx is None:
            signal_idx = self.np_random.integers(0, 3)
        self.signal_idx = int(signal_idx)
        self.current_goal_pose = np.array([self.goal_poses[self.signal_idx][0], self.goal_poses[self.signal_idx][1], 0])

        # Local controller params.
        self.k_p, self.k_v = 100, 20    # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # legcay set_state for data compatibility
        self.legacy = legacy

        self.signal_occured = False
        # agent_pos, block_pos, block_angle
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3,render_size,render_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=0,
                high=ws,
                shape=(2,),
                dtype=np.float32
            )
        })
        self.box_color_dict = {"box1":COLORS["MAGENTA"], "box2":COLORS["ORANGE"], "box3":COLORS["BROWN"]}
        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([ws,ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )
 
        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None
        self.blink_counter = 0
        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state

        self.max_score = 50 * 100
        self.success_threshold = 0.95 
        self.agent = {}
        self.boxes = []
        self.current_box = pymunk.Body()

    def reset(self):
        self._setup()
        self.blink_counter = 0  # Reset blink_counter here so it flashes at the start of each episode
        observation = self._get_obs()
        return observation
    
    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')    # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(self, position:tuple, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        self.space.add(body, shape)
        return body

    def add_box(self, position:tuple, height, width):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color('LightSlateGray')
        self.space.add(body, shape)
        return body
    
    def _get_obs(self):
        img = self.render_frame(mode='rgb_array')

        agent_pos = np.array(self.agent.position)
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        
        obs = {
            'image': img_obs,
            'agent_pos': agent_pos
        }
        return obs
    
    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 70))
        body = pymunk.Body(mass, inertia)
        body.position = pose[:2].tolist()
        body.angle = 0

        # Add a shape to the body
        shape = pymunk.Poly.create_box(body, (50, 70))  # Create a box shape
        return body

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()
        self.red_done = False
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)
        self.current_box = self.add_box((250, 400), 50,70)
        self.agent = self.add_circle((250, 350),15)
    
    def _get_info(self):
        info = {
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            'block_pose': np.array(list(self.current_box.position) + [self.current_box.angle]),
            'goal_pose': self.current_goal_pose,
            }
        return info

    def change_circle_color(self, circle_idx, screen):
        pygame.draw.circle(screen,self.box_color_dict[f"box{circle_idx+1}"],
                              (self.signal_circle_poses[circle_idx][0], self.signal_circle_poses[circle_idx][1]),
                               self.signal_circle_radius)

    def render_frame(self,mode):
        flash_color = False
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()
        if self.blink_counter < 80:
            flash_color = True
            

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)


        # Draw the 3 goal poses.
        for i, goal_pose in enumerate(self.goal_poses):
            rect = pygame.Rect(goal_pose[0] - 25, goal_pose[1] - 35, 50, 70) #Expects the coordinates of the top left corner
            pygame.draw.rect(canvas, self.box_color_dict[f"box{i+1}"], rect=rect)
            pygame.draw.circle(canvas,COLORS["BLACK"],
                              (self.signal_circle_poses[i][0], self.signal_circle_poses[i][1]),
                               self.signal_circle_radius)
        if flash_color:
            self.change_circle_color(self.signal_idx, canvas)

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is already ticked during in step for "human"


        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action and (self.latest_action is not None):
            action = np.array(self.latest_action)
            coord = (action / 512 * 96).astype(np.int32)
            marker_size = int(8/96*self.render_size)
            thickness = int(1/96*self.render_size)
            cv2.drawMarker(img, coord,
                color=(255,0,0), markerType=cv2.MARKER_CROSS,
                markerSize=marker_size, thickness=thickness)
        self.blink_counter += 1
        return img

    def render(self, mode):
        return self.render_frame(mode)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        def act(obs):
            del obs
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act
        return TeleopAgent(act)

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for _ in range(n_steps):
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt
                self.space.step(dt)

        goal_body = self._get_goal_pose_body(self.current_goal_pose)
        goal_geom,_ = pymunk_to_shapely(goal_body, self.current_box.shapes)
        block_geom,_ = pymunk_to_shapely(self.current_box, self.current_box.shapes)
        
        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        reward = np.clip(coverage / self.success_threshold, 0, 1)

        observation = self._get_obs()
        info = self._get_info()
        assert observation is not None, "env._get_obs() returned None"
        assert info is not None, "env._get_info() returned None"
        return observation, reward, reward >= 1.0, info

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)
        return [seed]
    
