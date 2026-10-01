import collections
import cv2
import gym
import math
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
import shapely.geometry as sg
from gym import spaces
from pymunk.vec2d import Vec2d

from envs.pusht.pymunk_override import DrawOptions

def pymunk_to_shapely(body, shapes, circle_segments=32):
    geoms = list()
    vertices = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):            
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
            vertices += [verts]
        elif isinstance(shape, pymunk.shapes.Circle):
            center = body.local_to_world(shape.offset)
            radius = shape.radius
            verts = []
            for i in range(circle_segments):
                angle = 2 * math.pi * i / circle_segments
                x = center.x + radius * math.cos(angle)
                y = center.y + radius * math.sin(angle)
                verts.append((x, y))
            verts.append(verts[0])  # Close the polygon
            
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

COLOR_PALETTE = [{"1":(128, 128, 128), "2": (255, 165, 0) }, {"2":(128, 128, 128), "1": (255, 165, 0)}]


class DrawerPusht(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,            
            legacy=False, 
            block_cog=None, damping=None,
            render_action=True,
            render_size=96,
            reset_to_state=None,
            signal_pos_idx = 0, 
            signal_color_idx = 0):
        self._seed = None
        self.seed()
        self.window_size = ws = 520   # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        self.red_done = False
        self.goal_poses = [(150,100), (150,350)]
        self.block_position = [(150, 120), (150, 400)]
        self.signal_circle_poses = [(450,80), (450,80), (450, 80)]
        self.signal_circle_radius = 20
        self.signal_pos_idx = int(signal_pos_idx)
        self.signal_color_idx = int(signal_color_idx)
        self.current_goal_pose = np.array([450,80, 0])
        self.chosen_palette = COLOR_PALETTE[self.signal_color_idx]
        self.box_contacted = [False, False]
        self.intermediate_circle_covered = False
        # Local controller params.
        self.k_p, self.k_v = 100, 20    # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # legcay set_state for data compatibility
        self.legacy = legacy
        self.correct_block_taken = False

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
        self.box_color_dict = {"box1":COLORS["MAGENTA"], "box2":COLORS["CYAN"], "box3":COLORS["BROWN"]}
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
        self.intermediate_circle= pymunk.Body()
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

    def add_box(self, position:tuple, height, width, color):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color(color)
        shape.outline_color = None  # Remove boundary/outline 
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
        # Add walls
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)
        self.boxes = [
            self.add_box(self.block_position[0], 50, 70, COLORS["MAGENTA"]),
            self.add_box(self.block_position[1], 50, 70, COLORS["CYAN"])
        ]
        self.current_box = self.boxes[self.signal_pos_idx]
        self.agent = self.add_circle((450, 350),15)
        self.intermediate_circle = self.add_circle((450, 200),20)
    
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

    def check_change_color(self):
        for i, goal_pose in enumerate(self.goal_poses):
            if (abs(self.agent.position[0] - goal_pose[0]) < 60.0 and abs(self.agent.position[1] - goal_pose[1]) < 60.0):
                shape = next(iter(self.boxes[i].shapes))
                shape.color = pygame.Color(self.chosen_palette[f"{i+1}"])
                self.intermediate_circle_covered = False
            else:
                shape = next(iter(self.boxes[i].shapes))
                if abs((shape.body.position - self.agent.position).length) > 60 and not self.box_contacted[i]:
                    shape.color = pygame.Color(self.box_color_dict[f"box{i+1}"])
                else:
                    self.box_contacted[i] = True
                    shape.color = pygame.Color(self.chosen_palette[f"{i+1}"])
                    self.correct_block_taken = True
                    self.intermediate_circle_covered = True
    
    def render(self, mode):
        return self.render_frame(mode)
    
    def render_frame(self,mode, flash_color=False):
        del flash_color
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)
        rect = pygame.Rect(self.signal_circle_poses[0][0] - 25, self.signal_circle_poses[0][1] - 35, 50, 70) #Expects the coordinates of the top left corner
        pygame.draw.rect(canvas, self.chosen_palette[f"{self.signal_pos_idx+1}"], rect=rect)
        for i, goal_pose in enumerate(self.goal_poses):
            s = pygame.Surface((150, 200), pygame.SRCALPHA)  # Create a surface with alpha channel
            s.fill((*self.box_color_dict[f"box{i+1}"], 250))  # Add alpha value (128 for 50% transparency)
            canvas.blit(s, (goal_pose[0] - 75, goal_pose[1] - 100))

        self.check_change_color()

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
        return img

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
        reward = 0.0
        if action is not None:
            self.latest_action = action
            for _ in range(n_steps):
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt
                self.space.step(dt)

        goal_body = self._get_goal_pose_body(self.current_goal_pose)
        goal_geom,_ = pymunk_to_shapely(goal_body, self.current_box.shapes)
        block_geom,_ = pymunk_to_shapely(self.current_box, self.current_box.shapes)

        intermidiate_circle_geom ,_ = pymunk_to_shapely(self.intermediate_circle, self.intermediate_circle.shapes)
        agent_geom ,_ = pymunk_to_shapely(self.agent, self.agent.shapes)

        if not self.intermediate_circle_covered:
            intersection_intermediate_area = agent_geom.intersection(intermidiate_circle_geom).area
            coverage_intermediate = intersection_intermediate_area / agent_geom.area
            
            if coverage_intermediate > 0.75:
                self.intermediate_circle_covered = True 
                reward = 0.5
            else:
                self.intermediate_circle_covered = False
                reward = 0.0
        else:
            intersection_area = goal_geom.intersection(block_geom).area
            goal_area = goal_geom.area
            coverage = intersection_area / goal_area
            reward = np.clip(coverage / self.success_threshold, 0, 1)

        observation = self._get_obs()
        info = self._get_info()
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
