import math
from random import randint
import random
import gym
import scipy.misc
from gym.utils import seeding
from gym import spaces
import pygame
from collections  import deque
import time

import random
from functools import reduce
from collections import deque

DEGREE = 0.017453292519943295
EMPTY = 0.0
FULL = 255.0
SSL = 4 ## seven segment line

#TODO change maximum steps
config = {
    "channel_numbers": 4,
    "maximum_steps": 45,
    "draw_channel": 0,
    "turtle_channel": 1,
    "target_channel": 2,
    "helper_channel": 3,
    "start_channel": 4,
    "draw_reward": 2,
    "prospect_reward": [150],
    "draw_punish": 12,
    "time_punish": 1,
    "use_gpu_array": False,
    "target_line_width": 4,
    "turtle_line_width": 2,
    "end_state_reward": 5000,
    "recent_actions_history": 30,
    "rotate_degree": 90,
    "polygon_size": 4,
    "prospect_width": 5,
    "prospect_length":[10],
    "forward_distance": 5,
    "draw": True,
    "draw_random_start": False,
    "edge_detection": False,
    "select_start_point": False
}

if config["use_gpu_array"]:
    import minpy.numpy as np #NOTE it does not support rendering via scipy.misc
else:
    import numpy as np


class Graphics:
    def __init__(self, width=256, height=256, channels=4):

        self.pixels = np.zeros((width, height, channels))

    def reset(self):
        self.clear([range(config['channel_numbers'])])
        self.target_p =0
        self.draw_p = 0
        self.draw_wrong_p = 0
        self.lsp = deque([[], []], 2)
        self.tlsp = []
        self.nlsp = False
        self.last_line = [(0,0), (0,0)]


    def clear(self, channels=[0]):
        for ch in channels:
            self.pixels[:, :, ch] = 0.



    def plot(self, x, y, channels=[0], inverse=False):
        if x < 0 or y < 0:
            return
        repeated = False
        data = EMPTY if inverse else FULL
        if channels == [config['draw_channel']]:
            if self.pixels[x, y, config['target_channel']] == FULL and self.pixels[x, y, config['draw_channel']] == FULL:
                if self.pixels[x,  y, config['helper_channel']] == EMPTY:
                    repeated = True
                self.pixels[x, y, config['helper_channel']]=FULL
            self.tlsp.append((x, y))

        self.pixels[x, y, channels] = data
        return repeated


    def draw_polygon(self, points):
        start = points[0]
        self.target_p = 0
        for p in points[1:]:
            _, temp, _, _ = self.line(*start, *p, config['target_channel'], line_width=config['target_line_width'])
            self.target_p += temp
            start = p
        _, temp, _, _ = self.line(*p, *points[0], config['target_channel'], line_width=config['target_line_width'])
        self.target_p += temp

    def draw_arc(self, center, radius, degree, channels, fill):
        for i in range(degree):
            x = int(radius*math.cos(i*DEGREE))
            y = int(radius*math.sin(i*DEGREE))
            if(fill):
                self.line(center[0], center[1], center[0]+x, center[1]+y,
                       channels,
                       False, config['target_line_width'], False)
            else:
                self._plot(center[0]+x, center[1]+y,
                       channels,
                       False, config['target_line_width'], False)

    def rect(self, x0, y0, width, height, channels=config['target_channel']):
        p= 0
        _, temp, _, _ = self.line(x0, y0, x0 + width, y0, channels, line_width=config['target_line_width'])
        p+= temp
        _, temp, _, _ = self.line(x0, y0, x0, y0 + height, channels, line_width=config['target_line_width'])
        p+= temp
        _, temp, _, _ = self.line(x0 + width, y0, x0 + width, y0 + height, channels, line_width=config['target_line_width'])
        p+= temp
        _, temp, _, _ = self.line(x0, y0 + height, x0 + width, y0 + height, channels, line_width=config['target_line_width'])
        p+= temp
        self.target_p = p

    def calc_point_reward(self, x, y):
        try:
            if self.pixels[x, y, config['target_channel']]==FULL\
                and self.pixels[x, y, config['draw_channel']]==EMPTY:
                return config['draw_reward']
            elif self.pixels[x, y, config['target_channel']]==FULL\
                and self.pixels[x, y, config['draw_channel']]==FULL:
                    if self.pixels[x,y, config['helper_channel']]==FULL:
                        return -1*config['draw_punish']
                    else:
                        return -0.001*config['draw_punish']
            else:
                return -1*config['draw_punish']
        except IndexError:
            return -1 * config['draw_punish']

    def is_in_points(self, x, y, container):
        return any((x, y) in l for l in container)


    def calc_base(self, channels, line_width, calc_reward):
        if config["helper_channel"] in channels and calc_reward == False:
            return 0
        elif config['turtle_channel'] in channels:
            return 0
        else:
            return -line_width


    def _plot(self, x, y, channels, inverse, line_width, calc_reward):

        reward, tr = 0, 0
        good_points_counter, bad_points_counter, tpc = 0, 0, 0
        line_width = int(line_width/2)
        repeated = True
        base = self.calc_base(channels, line_width, calc_reward)

        for d in range(base, line_width):
            for d2 in range(base, line_width):
                try:
                    if calc_reward:
                        if self.is_in_points(x + d, y + d2, [self.tlsp]):
                            continue
                        if self.nlsp and self.is_in_points(x + d, y + d2, self.lsp):
                            continue
                        temp = self.calc_point_reward(x + d, y + d2)
                        reward += temp
                        if temp>0:
                            good_points_counter += 1
                        elif self.pixels[x + d, y + d2, config['draw_channel']] == EMPTY:
                            bad_points_counter += 1
                    elif self.pixels[x+d, y+d2, config['target_channel']] == EMPTY:
                        good_points_counter += 1
                    plot_out = self.plot(x + d, y + d2, channels, inverse)
                    repeated = repeated and plot_out
                except IndexError as e:
                    continue
        return reward, good_points_counter, bad_points_counter, repeated

    def draw_number(self, origin, num, SSL, chan=[config['helper_channel'], config['draw_channel']], clear = False):
        digits = []
        if(num==0):
            digits=[0]
        else:
            while(num!=0):
                r = num%10
                digits.append(int(r))
                num = (num-r)/10
            digits.reverse()

        for index, d in enumerate(digits):
            self.draw_digit([origin[0]+index*(SSL[0]+1), origin[1]], d, SSL, chan, clear)

    def draw_digit(self, origin, num, SSL, chan, clear = False, simulate = False):
        ss = []
        if(num==0):
            ss = [0, 1, 4, 6, 5, 2]
        elif(num==1):
            ss = [1,4]
        elif(num==2):
            ss = [0, 2, 3, 4, 6]
        elif(num==3):
            ss = [0, 2, 3, 5, 6]
        elif(num==4):
            ss = [1, 3, 2, 5]
        elif(num==5):
            ss = [0, 1, 3, 5, 6]
        elif(num==6):
            ss = [0, 1, 4, 6, 5, 3]
        elif(num==7):
            ss = [0, 2, 5]
        elif(num==8):
            ss = [0, 1, 2, 3, 4, 5, 6]
        elif(num==9):
            ss = [0, 1, 2, 3, 5, 6]
        if simulate == True:
            return ss
        else:
            self.draw_digit_line(origin, ss, SSL, chan, clear)


#    ssl[0]
# ------ 0
# |1    |2    ssl[1]
# ------ 3
# |4    |5    ssl[2]
# ------ 6
#    ssl[0]

    def give_line_num_points(self, a, origin, SSL):
        if(a==0):
            return [(origin[0], origin[1]), (origin[0]+SSL[0], origin[1])]
        elif(a==1):
            return [(origin[0], origin[1]), (origin[0], origin[1]+SSL[1])]
        elif(a==2):
            return [(origin[0]+SSL[0], origin[1]), (origin[0]+SSL[0], origin[1]+SSL[1])]
        elif(a==3):
            return [(origin[0], origin[1]+SSL[1]), (origin[0]+SSL[0], origin[1]+SSL[1])]
        elif(a==4):
            return [(origin[0], origin[1]+SSL[1]), (origin[0], origin[1]+SSL[1]+SSL[2])]
        elif(a==5):
            return [(origin[0]+SSL[0], origin[1]+SSL[1]), (origin[0]+SSL[0], origin[1]+SSL[1]+SSL[2])]
        elif(a==6):
            return [(origin[0], origin[1]+SSL[1]+SSL[2]), (origin[0]+SSL[0], origin[1]+SSL[1]+SSL[2])]

    def draw_digit_line(self, origin, array, SSL, chan, clear):
        while(len(array)>0):
            temp = 0
            a = array[0]
            line_width = config['target_line_width']
            if(chan!=config['target_channel']):
                line_width = int(config['target_line_width']/2)
            points = self.give_line_num_points(a, origin, SSL)
            _, temp, _, _ = self.line(points[0][0], points[0][1], points[1][0], points[1][1], chan, clear, line_width)
            del array[0]
            if chan == config['target_channel']:
                self.target_p += temp

    def check_repeated_line(self, first, second):
        try:
            self.last_line.index(first)
            self.last_line.index(second)
            return False
        except Exception as e:
            return True

    def line(self, x0, y0, x1, y1, channels=[0], inverse=False, line_width=config['target_line_width'], calc_reward=False):
        deltax = x1 - x0
        deltay = y1 - y0

        if(channels==[config["draw_channel"]]):
            self.nlsp = self.check_repeated_line((x0, y0), (x1, y1))

        sgn = 1 if deltay >= 0 else -1
        if type(channels) != list:
            channels = [channels]
        repeated = True
        reward = 0
        gpc, bpc = 0, 0
        if deltax != 0:
            deltaerr = abs(deltay / deltax)
            error = 0.
            y = y0
            if x0 > x1:
                x1, x0 = x0, x1
                y = y1
                sgn *= -1
            for x in range(x0, x1):
                tr, tgpc, tbpc, rep = self._plot(x, y, channels, inverse, line_width, calc_reward)
                reward += tr
                gpc += tgpc
                bpc += tbpc
                error += deltaerr
                repeated = rep and repeated
                while error >= 0.5:
                    y += sgn
                    tr, tgpc, tbpc, rep  = self._plot(x, y, channels, inverse, line_width, calc_reward)
                    reward += tr
                    gpc += tgpc
                    bpc += tbpc
                    error -= 1.
                    repeated = rep and repeated
        else:
            if y1 < y0:
                y1, y0 = y0, y1
            for y in range(y0, y1):
                tr, tgpc, tbpc, rep = self._plot(x0, y, channels, inverse, line_width, calc_reward)
                reward += tr
                gpc += tgpc
                bpc += tbpc
                repeated = rep and repeated
        if (channels == [config["draw_channel"]]):
            self.last_line = [(x0, y0), (x1, y1)]

        return reward, gpc, bpc, repeated


class Env(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, width=128, height=128):
        self._seed()
        self.state = None
        self.maximum_steps = config['maximum_steps']

        if config['draw']==True:
            if config['select_start_point']:
                self.action_space = spaces.Discrete(6)
            else:
                self.action_space = spaces.Discrete(4)
            config['maximum_steps'] = 55
            if config['edge_detection']:
                config['maximum_steps'] = 6
            width = 40
            height = 64
        else:
            self.action_space = spaces.Discrete(3)
            config['maximum_steps'] = 30
            width = 100
            height = 110

        self.screen_width = width
        self.screen_height = height
        self.graphics = Graphics(width, height)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width, 3))
        self.display = None
        self.coordinate = [5, int(self.width/18)]
        self.coordinate2 = [17, int(self.width/18)]
        self.coordinate3 = [29, int(self.width / 18)]

    def _select_polygon_points(self, num):
        x = self.np_random.randint(self.screen_width/2-6, self.screen_width/2+20, 1)
        y = self.np_random.randint(self.screen_width/2-3, self.screen_width/2+5, 1)
        start = (x[0], y[0])
        base = 180/config['rotate_degree']
        base_angle = math.pi / base
        angle_options = (np.arange(base - 2) + 1) * base_angle
        angle = 0
        rotate_degree = self.np_random.choice(angle_options , 3)
        size = np.array(np.random.choice(range(3, 5), 3))*5
        #size = [3,3,4]
        points = []
        points.append(start)
        for i in range(num - 1):
            angle+=rotate_degree[i]
            next_x = int(start[0] + size[i] * math.cos(angle))
            next_y = int(start[1] + size[i] * math.sin(angle))
            next_point = (next_x, next_y)
            points.append(next_point)
            start = next_point
        dist = points[0][0] - points[-1][0]
        if dist<0:
            next_y  = math.tan(angle_options[0])*dist
        else:
            next_y  = math.tan(-angle_options[0])*dist
        next_point = (points[0][0], int(points[-1][1] + next_y))
        points.append(next_point)
        return points

    def count_not_target_points(self, idxs):
        answer = []
        for idx in idxs:
            partition = self.graphics.pixels[idx[0]: idx[1], idx[2]:idx[3], config['target_channel']]
            answer.append(np.prod(partition.shape) - np.count_nonzero(partition==FULL))
        return np.count_nonzero(np.array(answer)<5)

    def initial_point_reward(self, loc):
        line_width = int(config['target_line_width']/2)
        distance = config['forward_distance']
        long_width = line_width + distance
        left_idx  = [-line_width, long_width, -line_width, line_width]
        right_idx  = [-long_width, line_width, -line_width,line_width]
        down_idx = [-line_width, line_width, -line_width, long_width]
        up_idx = [-line_width, line_width, -long_width, line_width]
        base = np.array([loc[0], loc[0], loc[1], loc[1]])
        idxs = np.array([left_idx, right_idx, down_idx, up_idx])
        for i in range(len(idxs)):
            idxs[i] = np.minimum(np.maximum(idxs[i]+base, 0), [self.width, self.width, self.height, self.height])
        target_direction = self.count_not_target_points(idxs)
        if target_direction == 0:
            return -300
        elif target_direction == 1:
            return 300
        elif target_direction == 2:
            return 0
        elif target_direction>2:
            return -200
        # return 0

#TODO make repeat more annoying
#TODO make all rewards negative

    def fill_the_point(self, ps, clear = False):
        width = int(config['target_line_width']/2)
        for p in ps:
            self.graphics.pixels[p[0]-width:p[0]+width, p[1]-width: p[1]+width, config['helper_channel']]= EMPTY if clear == True else FULL

    def calc_available_starting_points(self, num):
        if num == 1:
            return [0, 4]
        elif num == 2:
            return range(6)
        elif num == 3:
            return range(6)
        elif num == 4:
            return [0, 1, 2, 3, 5]
        elif num == 5:
            return range(6)
        elif num == 6:
            return range(6)
        elif num == 7:
            return [0, 1, 5]
        elif num == 8:
            return range(6)
        elif num == 9:
            return range(6)

    def reset(self):
        self.repeated_forwards = 0
        self.circular = 0
        self.basic_coverage = 0
        self.graphics.reset()
        self.start_candidates = []
        self.rotate_right = 0
        self.rotate_left = 0
        self.recent_actions = []
        self.recent_rewards = deque(maxlen=10)
        self.recent_rotate_number = 0
        self.pen = True
        self.wrong_forward = False
        self.wrong_forward_numbers = 0
        self.consec_repeat = 1
        self.first_forward = True
        self.before_pen = False if config['draw'] else True
        self.remaining_steps = self.maximum_steps
        self.total_repeat = 1
        self.before_draw = config['select_start_point']

        if config['draw']:
            x = 10
            y = 20
        else:
            x = np.random.choice(range(3, 16))*5
            y = np.random.choice(range(6, 13))*5
        self.origin = [x, y]

        size = np.array(np.random.choice(range(3,5), 3)*5).astype(int)
        #size = [15, 15, 20]
        prob = [0.9]*9
        prob[1] = 0.2
        target_num = np.random.choice(range(1, 10), size = 1)
        origin = self.origin

        self.corner_locations = [origin,
                       [origin[0]+size[0], origin[1]],
                       [origin[0], origin[1]+size[1]],
                       [origin[0]+size[0], origin[1]+size[1]],
                       [origin[0], origin[1]+size[1]+size[2]],
                       [origin[0]+size[0], origin[1]+size[1]+size[2]]]




        self.graphics.draw_number(origin, target_num[0], size, chan=config['target_channel'])
        #self.graphics.draw_number(self.coordinate, self.rotate_right, [SSL] * 3)
        #self.graphics.draw_number(self.coordinate2, self.rotate_left, [SSL] * 3)
        #self.graphics.draw_number(self.coordinate3, self.recent_rotate_number, [SSL] * 3)

        #random polygon
        # polygon_points = self._select_polygon_points(config["polygon_size"])
        # self.graphics.draw_polygon(polygon_points)

        #random rectangle
        # point = self.np_random.randint(10, self.screen_width/2, 2)
        # self.graphics.rect(*point, size[0], size[1])
        if config['draw'] == True:
            self.no_repeat_boundary = 0
            if config['edge_detection'] or config['draw_random_start']:
                #if config['edge_detection']:
                #TODO make it random
                self.angle = np.random.choice([0, math.pi/2, -math.pi/2, math.pi])
                lines = self.graphics.draw_digit(origin, target_num, SSL, 0, simulate = True)
                line = np.random.choice(lines, 1)
                ps = (np.array(self.graphics.give_line_num_points(line, origin, size))/5).astype(int)
                self.tx = np.random.choice(range(ps[0][0], ps[1][0]+1))*5
                self.ty = np.random.choice(range(ps[0][1], ps[1][1]+1))*5
                self.ot = [self.tx, self.ty]
            else:
                 self.angle = 0
                 options = self.calc_available_starting_points(target_num)
                 start = np.array(self.corner_locations)[options]
                 available_points =  sorted(np.random.choice(range(len(start)), size = np.random.randint(2, len(start)+1), replace = False))
                 self.start_point = start[available_points]
                 #self.start_point = start
                 self.tx, self.ty = self.start_point[np.random.randint(len(self.start_point))]
                 if config['select_start_point']:
                    self.fill_the_point(self.start_point)
            #self.no_repeat_boundary = max(size[0], size[1], size[2])*2/5
        else:
            self.angle = 0
            x_opt = [5, 10, 15]
            self.tx = np.random.choice(x_opt, p = [0.6, 0.2, 0.2])
            #self.tx = x_opt[0]
            y_opt = [20, 25, 30]
            if self.tx == x_opt[0]:
                self.ty = np.random.choice(y_opt)
            else:
                self.ty = 20
            #self.ty = y_opt[0]
            

        self.graphics.draw_number(self.coordinate, self.rotate_right, [SSL] * 3)
        self.graphics.draw_number(self.coordinate2, self.rotate_left, [SSL] * 3)
        self.graphics.draw_number(self.coordinate3, self.recent_rotate_number, [SSL] * 3)

        if config['select_start_point']==False:
            self._draw_turtle()
        self._done = False
        self.initial_distance = self.calc_distance(self.tx, self.ty)
        self.total_diff = 0
        self.maximum_moving_step = int(math.sqrt((self.initial_distance**2)/2)/config['forward_distance']*1.3)*2+3
        # num = self.np_random.randint(3, 8)
        # self._draw_polygon(num)

        return np.uint8(self.graphics.pixels)

    def _forward(self, distance, pen = True, simulate = False):
        self.graphics.lsp.append(self.graphics.tlsp)
        self.graphics.tlsp = []
        reward, p, repeated = 0, 0, False
        cx = int(distance * math.cos(self.angle))
        cy = int(distance * math.sin(self.angle))
        if (self.tx + cx >= (self.screen_width+5) or self.ty + cy >= (self.screen_height+5) or self.ty + cy < SSL-5 or self.tx + cx < -5):
            self._rotate(math.pi, False)
            cx = int(distance * math.cos(self.angle))
            cy = int(distance * math.sin(self.angle))
        if pen and simulate == False:
            reward, gp, bp, repeated = self.graphics.line(self.tx, self.ty, self.tx + cx, self.ty + cy,
                                        channels=[config['draw_channel']], line_width=config['target_line_width'],
                                        calc_reward=True)
            self.wrong_forward = bp>0
            self.graphics.draw_p += gp
            self.graphics.draw_wrong_p += bp
        else:
            self.wrong_forward = False

        if simulate == False:
            self.tx += cx
            self.ty += cy
        return reward, repeated, (self.tx + cx, self.ty + cy)

    def _rotate(self, angle, draw=True):
        self.angle += angle
        if self.angle < 0.:
            self.angle += 2 * math.pi
        elif self.angle > 2 * math.pi:
            self.angle -= 2 * math.pi

    def _draw_turtle(self, clear=False):
        cx = int(12 * math.cos(self.angle))
        cy = int(12 * math.sin(self.angle))
        p0 = (self.tx + cx, self.ty + cy)
        cx = int(4 * math.cos(self.angle + math.pi / 2.))
        cy = int(4 * math.sin(self.angle + math.pi / 2.))
        p1 = (self.tx + cx, self.ty + cy)
        cx = int(4 * math.cos(self.angle - math.pi / 2.))
        cy = int(4 * math.sin(self.angle - math.pi / 2.))
        p2 = (self.tx + cx, self.ty + cy)
        self.graphics.line(*p0, *p1, [config['turtle_channel']], clear, line_width=config['turtle_line_width'])
        self.graphics.line(*p0, *p2, [config['turtle_channel']], clear, line_width=config['turtle_line_width'])
        self.graphics.line(*p1, *p2, [config['turtle_channel']], clear, line_width=config['turtle_line_width'])

    def render(self, mode='human', close=False):
        if(self.display==None):
            pygame.init()
            display = pygame.display.set_mode((self.width, self.height))
        temp = self.graphics.pixels[:,:,1] + self.graphics.pixels[:,:,3]
        data = np.uint8(np.stack((self.graphics.pixels[:,:,0], temp, self.graphics.pixels[:,:,2]), axis = 2))
        surf = pygame.surfarray.make_surface(data)
        display.blit(surf, (0, 0))
        pygame.display.update()
        return


    def _is_done(self):
        coverage = float(self.graphics.draw_p)/float(self.graphics.target_p)
        if config['edge_detection'] == False and coverage > 0.99:
            end_reward_coeff = 1 if self.graphics.draw_wrong_p < (self.graphics.target_p/10) else 0
            #end_reward_coeff2 = (self.graphics.draw_p) / (self.graphics.draw_wrong_p*1.5 + self.graphics.draw_p)
            if config['draw_random_start']:
                self.graphics.pixels[:,:,  config['helper_channel']] = EMPTY
                self.fill_the_point(self.start_candidates)
                #print("candidate", self.start_candidates)
            #coeff = 1+6*self.remaining_steps/self.maximum_steps
            #print("COEEFFF", coeff)
            #print("total_repeat", self.total_repeat)
            return True, max(((config['end_state_reward']-self.total_repeat*200)/self.total_repeat),200) * end_reward_coeff
        if self.before_pen == True and self.graphics.pixels[min(self.tx, self.width-1), min(self.ty, self.height-1), config['target_channel']]==FULL:
            self.fx = self.origin[0] - 10
            self.sx = self.origin[0] + 30
            self.fy = self.origin[1] - int(config['target_line_width']/2)
            self.sy = self.origin[1] + 40
            self.pixels = self.graphics.pixels[self.fx:sx, fy:sy, :]
            temp = self.graphics.pixels[:,:,1] + self.graphics.pixels[:,:,3]
            data = np.uint8(np.stack((self.graphics.pixels[:,:,0], temp, self.graphics.pixels[:,:,2]), axis = 2))
            shape = data[fx:sx, fy:sy, :]
            header = data[0:40, 0:18, :]
            cut = np.hstack((header, shape))
            scipy.misc.imsave('1.jpg', np.flip(cut, axis = 0))
            return True, config['end_state_reward']
        if config["draw"] and  (config["edge_detection"] or config['draw_random_start']):
            #right_reward = self.calc_prospective_reward(self.angle+config['rotate_degree']*DEGREE)
            #left_reward =  self.calc_prospective_reward(self.angle-config['rotate_degree']*DEGREE)
            #print(right_reward, left_reward)
            #if right_reward>0 or left_reward>0:
            #   return True, config['end_state_reward']
            if len(self.recent_actions)>0 and self.recent_rewards[-1]>(-config['time_punish']) and self.recent_actions[-1]==0:
                right_reward = self.calc_prospective_reward(self.angle+config['rotate_degree']*DEGREE)
                left_reward =  self.calc_prospective_reward(self.angle-config['rotate_degree']*DEGREE)
                forward_reward = self.calc_prospective_reward(self.angle)
                if right_reward>0 or left_reward>0:
                    self.start_candidates.append((self.tx, self.ty))
                if right_reward<=0 and left_reward<=0 and forward_reward<=0:
                    self.start_candidates.append((self.tx, self.ty))

        if self.remaining_steps <= 0:
            return True, 0
        return False, 0

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

#change logger
# make lines thicker

    def calc_prepend_points(self, p, angle, width):
        p1y = math.sin(angle + math.pi / 2) * width + p[1]
        p1x = math.cos(angle + math.pi / 2) * width + p[0]
        p2y = math.sin(angle - math.pi / 2) * width + p[1]
        p2x = math.cos(angle - math.pi / 2) * width + p[0]
        return ((int(p1x), int(p1y)), (int(p2x), int(p2y)))

    def calc_prospective_reward(self, angle=None):
        if angle == None:
            angle = self.angle
            prospect_width = config['prospect_width']
        else:
            prospect_width = config['target_line_width']/2
        fps = self.calc_prepend_points((self.tx+math.cos(angle)*config['target_line_width']/2, self.ty + math.sin(angle)*config['target_line_width']/2), angle, prospect_width)
        prospect = config['prospect_length']
        npoint = []
        for pr in prospect:
            npoint.append((math.cos(angle)*pr+self.tx, math.sin(angle)*pr+self.ty))
        sps = []
        for nps in npoint:
            sps.append(self.calc_prepend_points(nps, angle, prospect_width))
        #TODO does not work properly for fractional degrees
        prospect_reward = []
        for index, point in enumerate(sps):
            xs = [fps[0][0], fps[1][0], point[0][0], point[1][0]]
            ys = [fps[0][1], fps[1][1], point[0][1], point[1][1]]
            minx = max(min(xs), 0)
            maxx = min(max(xs), self.screen_width)
            miny = max(min(ys), 0)
            maxy = min(max(ys), self.screen_height)
            rectangle_view = np.array(self.graphics.pixels[minx:maxx, miny:maxy,:])
            target_points = np.logical_and(rectangle_view[:,:,config['target_channel']]==FULL, rectangle_view[:,:,config['draw_channel']]==EMPTY)
            total_points = 2*prospect_width*prospect[index]
            prospect_reward.append(np.count_nonzero(target_points)/total_points)
        rewards = prospect_reward * np.array(config['prospect_reward'])
        rewards *= ((1 + self.coverage - self.basic_coverage)*self.turn_effect)**3
        return sum(rewards)


    def calc_distance(self, x, y):
        return math.sqrt((x - self.origin[0])**2 + (y - self.origin[1])**2)

    def calc_draw_reward(self, index, repeated, one_step_reward):
        reward = 0
        if repeated and (self.maximum_steps-self.remaining_steps)<=self.no_repeat_boundary:
            reward -= 100
        if self.first_forward == True:
            if one_step_reward>35:
                one_step_reward = 41
            self.first_forward = False
        if self.wrong_forward:
            self.wrong_forward_numbers +=1
            self.basic_coverage = self.coverage
            one_step_reward = one_step_reward * self.wrong_forward_numbers
        elif self.wrong_forward == False:
            self.wrong_forward_numbers = 0
        if one_step_reward>0:
            one_step_reward *= ( index * 0.2 + 1 )
        if one_step_reward < 0 and repeated == True:
            self.consec_repeat += 1
            self.consec_repeat = min(self.consec_repeat, 2)
        else:
            #one_step_reward *= self.consec_repeat
            self.consec_repeat = 1
        reward += one_step_reward  * ((1 + self.coverage - self.basic_coverage)*self.turn_effect) ** 3
        return reward

    def calc_before_pen_reward(self, action, distance):
        reward, x, y = 0, 0, 0
        if action == 0:
            x, y = self.tx, self.ty
        elif action == 1 or action == 2:
            _, _, np = self._forward(config['forward_distance'], simulate = True)
            x, y = np
        dist2 = self.calc_distance(x, y)
        diff = distance - dist2
        if (self.maximum_steps - self.remaining_steps) > self.maximum_moving_step:
            reward -=  1000
        temp = 1
        if diff> 0:
            temp = 1.9
        reward -= int(dist2)**2/temp
        return 0


    def step(self, action_input):
        self._draw_turtle(True)
        reward, self.coverage = 0, 0

        if self.before_draw:
            act = action_input%len(self.start_point)
            self.tx, self.ty = self.start_point[act]
            self.before_draw = False
            self.fill_the_point(self.start_point, clear = True)
            #reward = self.initial_point_reward(loc)
        else:
            if action_input == 3:
                if config['draw']:
                    actions = [0,0,1,1,0,0]
                if config['edge_detection']:
                    actions = [1, 1]
            else:
                action_input %= 4
                actions = [action_input]
            self.wrong_forward = False
            rotate_right = self.rotate_right
            rotate_left = self.rotate_left
            recent_rotate = self.recent_rotate_number
            #TODO if null action is active it is required
            # if action!=0:

            for index, action in enumerate(actions):
                self.remaining_steps -= 1
                self.turn_effect = 1+ self.remaining_steps/self.maximum_steps
                #self.turn_effect = 1
                self.recent_actions.append(action)
                if len(self.recent_actions) > config['recent_actions_history']:
                    del self.recent_actions[0]
                self.recent_rotate_number = reduce(lambda x,y: x+1 if (y==1 or y ==2) else x, [0, *self.recent_actions])
                self.coverage = float(self.graphics.draw_p) / float(self.graphics.target_p)
                if config['draw']:
                    if config['edge_detection']:
                        config['time_punish'] = 200
                    else:
                        config['time_punish'] = 1
                else:
                    config['time_punish'] = 100
                reward -= config['time_punish']
                distance = self.calc_distance(self.tx, self.ty)
                if self._done == True:
                    return np.uint8(self.graphics.pixels), 0 , True, {'coverage': 0}
                if action == 0:
                    one_step_reward, repeated, _ = self._forward(config['forward_distance'], pen = not self.before_pen)
                    self.total_repeat += 1 if repeated else 0
                    if self.before_pen == False:
                       reward += self.calc_draw_reward(index, repeated, one_step_reward)
                    elif self.before_pen:
                        reward += self.calc_before_pen_reward(action, distance)
                    self.rotate_left = 0
                    self.rotate_right = 0
                elif action == 1 or action == 2:
                    consec_rotate = 0
                    if action ==1:
                        self.rotate_right += 1
                        consec_rotate = self.rotate_right
                        if rotate_left>0:
                            self.circular +=1
                            reward -= self.circular*2000
                        else:
                            self.circular = 0
                        self.rotate_left = 0
                        degree = DEGREE
                    if action == 2:
                        self.rotate_left += 1
                        consec_rotate = self.rotate_left
                        if rotate_right>0:
                            self.circular += 1
                            reward -= self.circular*2000
                        else:
                            self.circular = 0
                        self.rotate_right = 0
                        degree = -DEGREE
                    threshold = int(config['recent_actions_history']/2.5)
                    self._rotate(config['rotate_degree'] * degree)
                    consecutive_rotate_threshold = 2
                    if consec_rotate > consecutive_rotate_threshold:
                        reward -= (consec_rotate - consecutive_rotate_threshold) * 2000
                    if self.recent_rotate_number > threshold:
                        reward -= (self.recent_rotate_number- threshold)*2000
                    if not self.before_pen:
                        reward += self.calc_prospective_reward()
                    else:
                        reward += self.calc_before_pen_reward(action, distance)

            if self.rotate_right != rotate_right:
                self.graphics.draw_number(self.coordinate, rotate_right, [SSL] * 3, clear=True)
                self.graphics.draw_number(self.coordinate, self.rotate_right, [SSL] * 3)

            if self.rotate_left != rotate_left:
                self.graphics.draw_number(self.coordinate2, rotate_left, [SSL] * 3, clear=True)
                self.graphics.draw_number(self.coordinate2, self.rotate_left, [SSL] * 3)

            if self.recent_rotate_number != recent_rotate:
                self.graphics.draw_number(self.coordinate3, recent_rotate, [SSL] * 3, clear=True)
                self.graphics.draw_number(self.coordinate3, self.recent_rotate_number, [SSL] * 3)



            # elif action == 2:
            #     self._rotate(-10 * DEGREE)
            # elif action == 3:
            #     for _ in range(5):
            #         reward += self._forward(3)







            #         self._rotate(2 * DEGREE)
            # else:
                # if(self.pen_repeat==True and self.pen==True):
                    # reward -= 10000
                # if(self.pen==False):
                    # self.pen_repeat = True
                # self.pen = not self.pen

            self.recent_rewards.append(reward)
            self._done, tr = self._is_done()
            reward += tr
        self._draw_turtle(False)

        return np.uint8(self.graphics.pixels), reward, self._done, {'coverage': self.coverage}

import time

def main():
    current_milli_time = lambda: int(round(time.time() * 1000))
    env = Env(128, 128)

    for i in range(10):
        env.reset()
        start = current_milli_time()
        j = 0
        rt = 0
        total = 0
        while True:
            j+=1
            time.sleep(0.01)
            env.render()
            action = randint(0, 3)
            obs, r, done, dic = env.step(action)
            rt+= r
            if done:
                print("Done")
                break
            # if i == 0:
            #    env.render_on_file(str(j)+"_"+str(r))
        end = current_milli_time()
        total += rt
        # print(i, end-start, rt)
    # print(total/10.0)


if __name__ == '__main__':
    main()

'''graphics = Graphics(512, 512)
center = (256, 256)
for deg in range(0, 360):
    graphics.line(*center, int(center[0] + 45 * math.cos(deg / 180. * math.pi)),
                  int(center[1] + 45 * math.sin(deg / 180. * math.pi)))
scipy.misc.imsave('res.png', graphics.pixels)
'''
#
