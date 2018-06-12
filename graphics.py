import math
from random import randint
import gym
import scipy.misc
from gym.utils import seeding
from gym import spaces
import pygame
import time
import random
from functools import reduce
from collections import deque
import scipy

DEGREE = 0.017453292519943295
EMPTY = 0.0
FULL = 255.0
SSL = 4 ## seven segment line
config = {
    "channel_numbers": 4,
    "maximum_steps": 45,
    "draw_channel": 0,
    "turtle_channel": 1,
    "target_channel": 2,
    "helper_channel": 3,
    "draw_reward": 2,
    "prospect_reward": [100],
    "draw_punish": 12,
    "time_punish": 1,
    "use_gpu_array": False,
    "target_line_width": 4,
    "turtle_line_width": 2,
    "end_state_reward": 2000,
    "recent_actions_history": 30,
    "rotate_degree": 90,
    "polygon_size": 4,
    "prospect_width": 5,
    "prospect_length":[10],
    "forward_distance": 5
}

if config["use_gpu_array"]:
    import minpy.numpy as np
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
        self.pixels[:, :, channels] = 0.



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

    def draw_digit(self, origin, num, SSL, chan, clear = False):
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
        self.draw_digit_line(origin, ss, SSL, chan, clear)


#    ssl[0]
# ------ 0
# |1    |2    ssl[1]
# ------ 3
# |4    |5    ssl[2]
# ------ 6
#    ssl[0]
    def draw_digit_line(self, origin, array, SSL, chan, clear):
        while(len(array)>0):
            temp = 0
            a = array[0]
            line_width = config['target_line_width']
            if(chan!=config['target_channel']):
                line_width = int(config['target_line_width']/2)
            if(a==0):
                _, temp, _, _ = self.line(origin[0], origin[1], origin[0]+SSL[0], origin[1], chan, clear, line_width)
            elif(a==1):
                _, temp, _, _ = self.line(origin[0], origin[1], origin[0], origin[1]+SSL[1], chan, clear, line_width)
            elif(a==2):
                _, temp, _, _ = self.line(origin[0]+SSL[0], origin[1], origin[0]+SSL[0], origin[1]+SSL[1], chan, clear, line_width)
            elif(a==3):
                _, temp, _, _ = self.line(origin[0], origin[1]+SSL[1], origin[0]+SSL[0], origin[1]+SSL[1], chan, clear, line_width)
            elif(a==4):
                _, temp, _, _ = self.line(origin[0], origin[1]+SSL[1], origin[0], origin[1]+SSL[1]+SSL[2], chan, clear, line_width)
            elif(a==5):
                _, temp, _, _ = self.line(origin[0]+SSL[0], origin[1]+SSL[1], origin[0]+SSL[0], origin[1]+SSL[1]+SSL[2], chan, clear, line_width)
            elif(a==6):
                _, temp, _, _ = self.line(origin[0], origin[1]+SSL[1]+SSL[2], origin[0]+SSL[0], origin[1]+SSL[1]+SSL[2], chan, clear, line_width)
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
        if channels == [config["draw_channel"]]:
            self.last_line = [(x0, y0), (x1, y1)]

        return reward, gpc, bpc, repeated


class Env(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, width=128, height=128):
        self._seed()
        self.screen_width = width
        self.screen_height = height
        self.graphics = Graphics(width, height)
        self.width = width
        self.height = height
        self.state = None
        self.maximum_steps = config['maximum_steps']
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width, 3))
        self.display = None
        self.coordinate = [int(self.width*1/16), int(self.width/18)]
        self.coordinate2 = [int(self.width*5/16), int(self.width/18)]
        self.coordinate3 = [int(self.width*9/16), int(self.width / 18)]

    def _select_polygon_points(self, num):
        x = self.np_random.randint(self.screen_width/2-6, self.screen_width/2+20, 1)
        y = self.np_random.randint(self.screen_width/2-3, self.screen_width/2+5, 1)
        start = (x[0], y[0])
        base = 180/config['rotate_degree']
        base_angle = math.pi / base
        angle_options = (np.arange(base - 2) + 1) * base_angle
        angle = 0
        rotate_degree = self.np_random.choice(angle_options , 3)
        size = np.random.choice(range(3, 6), 3)*5
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

    def remove_helper_channel(self, input):
        temp = input[:, :, 1] + input[:, :, 3]
        return np.uint8(np.stack((input[:, :, 0], temp, input[:, :, 2]), axis=2))




    def reset(self):
        self.repeated_forwards = 0
        self.circular = 0
        self.basic_coverage = 0
        self.graphics.reset()
        self.angle = math.pi/2
        self.rotate_right = 0
        self.rotate_left = 0
        self.recent_actions = []
        self.recent_rotate_number = 0
        self.pen = True
        self.wrong_forward = False
        self.wrong_forward_numbers = 0
        self.consec_repeat = 1
        self.first_forward = True
        self.remaining_steps = self.maximum_steps
        self.before_draw = True

        x = np.random.choice(range(5, 40))
        y = np.random.choice(range(15, 20))
        # x = 5
        # y = 20
        origin = [x, y]
        size = np.array(np.random.choice(range(3,5), 3) * 5).astype(int)
        self.no_repeat_boundary = max(size[0], size[1], size[2])*2/5
        self.repeated_forward_counter = 0
        self.corner_locations = [origin,
                           [origin[0]+size[0], origin[1]],
                           [origin[0], origin[1]+size[1]+size[2]],
                           [origin[0]+size[0], origin[1]+size[1]+size[2]]]

        target_num = np.random.choice(range(1, 10), 1)
        self.before_draw_counter = 0
        self.graphics.draw_number(origin, target_num[0], size, chan=config['target_channel'])


        #random polygon
        # polygon_points = self._select_polygon_points(config["polygon_size"])
        # self.graphics.draw_polygon(polygon_points)

        #random rectangle
        # point = self.np_random.randint(10, self.screen_width/2, 2)
        # self.graphics.rect(*point, size[0], size[1])
        turtle_location = np.average(self.corner_locations, axis = 0)
        self.tx, self.ty = int(turtle_location[0]), int(turtle_location[1])
        self._draw_turtle()

        # self._draw_turtle()
        self._done = False


        # num = self.np_random.randint(3, 8)
        # self._draw_polygon(num)
        return self.remove_helper_channel(self.graphics.pixels)

    def _forward(self, distance):
        self.graphics.lsp.append(self.graphics.tlsp)
        self.graphics.tlsp = []
        reward, p = 0, 0
        cx = int(distance * math.cos(self.angle))
        cy = int(distance * math.sin(self.angle))
        if (self.tx + cx > (self.screen_width+5) or self.ty + cy > (self.screen_height+5) or self.ty + cy < SSL-5 or self.tx + cx < -5):
            self._rotate(math.pi, False)
            cx = int(distance * math.cos(self.angle))
            cy = int(distance * math.sin(self.angle))
        if self.pen:
            reward, gp, bp, repeated = self.graphics.line(self.tx, self.ty, self.tx + cx, self.ty + cy,
                                        channels=[config['draw_channel']], line_width=config['target_line_width'],
                                        calc_reward=True)

        self.tx += cx
        self.ty += cy
        self.wrong_forward = bp>0
        self.graphics.draw_p += gp
        self.graphics.draw_wrong_p += bp
        return reward, repeated

    def _rotate(self, angle, draw=True):
        self.angle += angle
        if self.angle < 0.:
            self.angle += 2 * math.pi
        elif self.angle > 2 * math.pi:
            self.angle -= 2 * math.pi

    def _draw_turtle(self, clear=False):
        cx = int(self.width/5 * math.cos(self.angle))
        cy = int(self.width/5 * math.sin(self.angle))
        p0 = (self.tx + cx, self.ty + cy)
        cx = int(self.width/15 * math.cos(self.angle + math.pi / 2.))
        cy = int(self.width/15 * math.sin(self.angle + math.pi / 2.))
        p1 = (self.tx + cx, self.ty + cy)
        cx = int(self.width/15 * math.cos(self.angle - math.pi / 2.))
        cy = int(self.width/15 * math.sin(self.angle - math.pi / 2.))
        p2 = (self.tx + cx, self.ty + cy)
        self.graphics.line(*p0, *p1, [config['turtle_channel']], clear, line_width=config['turtle_line_width'])
        self.graphics.line(*p0, *p2, [config['turtle_channel']], clear, line_width=config['turtle_line_width'])
        self.graphics.line(*p1, *p2, [config['turtle_channel']], clear, line_width=config['turtle_line_width'])

    def render(self, obs, mode='human', close=False):
        if(self.display==None):
            pygame.init()
            display = pygame.display.set_mode((self.width, self.height))
        # scipy.misc.imsave("sajad.png", obs.reshape(obs.shape[1], obs.shape[0]))
        surf = pygame.surfarray.make_surface(obs)
        display.blit(surf, (0, 0))
        pygame.display.update()
        return


    def _is_done(self):
        coverage = float(self.graphics.draw_p)/float(self.graphics.target_p)
        if coverage > 0.99:
            # print("coverage", coverage)
            # print("draw wrong number", self.graphics.draw_wrong_p)
            # print("forward", self.repeated_forward_counter)
            end_reward_coeff = 1 if self.graphics.draw_wrong_p < (self.graphics.target_p/10) else 0
            return True, (config['end_state_reward'] - self.repeated_forward_counter*175)* end_reward_coeff
        if self.remaining_steps == 0:
            # print("coverage", coverage)
            return True, 0
        return False, 0

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

#change logger
# make lines thicker

    def calc_prepend_points(self, p):
        width = config['prospect_width']
        p1y = math.sin(self.angle + math.pi / 2) * width + p[1]
        p1x = math.cos(self.angle + math.pi / 2) * width + p[0]
        p2y = math.sin(self.angle - math.pi / 2) * width + p[1]
        p2x = math.cos(self.angle - math.pi / 2) * width + p[0]
        return ((int(p1x), int(p1y)), (int(p2x), int(p2y)))

    def calc_prospective_reward(self):
        fps = self.calc_prepend_points((self.tx+math.cos(self.angle)*config['target_line_width']/2, self.ty + math.sin(self.angle)*config['target_line_width']/2))
        prospect = config['prospect_length']
        npoint = []
        for pr in prospect:
            npoint.append((math.cos(self.angle)*pr+self.tx, math.sin(self.angle)*pr+self.ty))
        sps = []
        for nps in npoint:
            sps.append(self.calc_prepend_points(nps))
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
            total_points = 2*config['prospect_width']*prospect[index]
            prospect_reward.append(np.count_nonzero(target_points)/total_points)
        rewards = prospect_reward * np.array(config['prospect_reward'])
        rewards *= (1 + self.coverage - self.basic_coverage)**3
        return sum(rewards)

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
            return -100
        elif target_direction == 1:
            return 100
        elif target_direction == 2:
            return 10
        elif target_direction>2:
            return -50
        # return 0




    def step(self, action_input):
        self._draw_turtle(True)
        reward = 0
        if self.before_draw == True:
            loc = self.corner_locations[action_input]
            self.tx, self.ty = loc[0], loc[1]
            self.before_draw_counter+=1
            self.before_draw = False
            self.graphics.draw_number(self.coordinate, self.rotate_right, [SSL] * 3)
            self.graphics.draw_number(self.coordinate2, self.rotate_left, [SSL] * 3)
            self.graphics.draw_number(self.coordinate3, self.recent_rotate_number, [SSL] * 3)
            reward = self.initial_point_reward(loc)
            self.coverage = 0
        else:

            if action_input==3:
                actions = [0, 1, 1, 0]
            else:
                actions = [action_input]
            self.wrong_forward = False
            rotate_right = self.rotate_right
            rotate_left = self.rotate_left
            recent_rotate = self.recent_rotate_number
            #TODO if null action is active it is required
            # if action!=0:


            for index, action in enumerate(actions):
                self.recent_actions.append(action)
                if (len(self.recent_actions) > config['recent_actions_history']):
                    del self.recent_actions[0]
                self.recent_rotate_number = reduce(lambda x,y: x+1 if (y==1 or y ==2) else x, [0, *self.recent_actions])
                self.coverage = float(self.graphics.draw_p) / float(self.graphics.target_p)
                if self._done == True:
                    return self.remove_helper_channel(self.graphics.pixels), 0 , True, {}

                reward -= config['time_punish']
                if action == 0:
                    temp_reward, repeated = self._forward(config['forward_distance'])
                    if repeated:
                        self.repeated_forward_counter+=1
                    if repeated and (self.maximum_steps-self.remaining_steps)<=self.no_repeat_boundary:
                        reward -= 100
                    if self.first_forward == True:
                        if(temp_reward>35):
                            temp_reward = 41+config['time_punish']
                        self.first_forward = False
                    if self.wrong_forward == True:
                        self.wrong_forward_numbers +=1
                        self.basic_coverage = self.coverage
                        temp_reward = temp_reward * self.wrong_forward_numbers
                    elif self.wrong_forward == False:
                        self.wrong_forward_numbers = 0
                    if temp_reward>0:
                        temp_reward *= ( index * 0.2 + 1 )
                    if temp_reward < 0 and repeated == True:
                        self.consec_repeat += 1
                        self.consec_repeat = min(self.consec_repeat, 2)
                    else:
                        temp_reward *= self.consec_repeat
                        self.consec_repeat = 1

                    reward += temp_reward  * (1 + self.coverage - self.basic_coverage) ** 3


                    self.rotate_left = 0
                    self.rotate_right = 0


                elif action == 1 or action == 2:
                    consec_rotate = 0
                    if action ==1:
                        self.rotate_right += 1
                        consec_rotate = self.rotate_right
                        if rotate_left>0:
                            self.circular +=1
                            reward -= self.circular*1000
                        else:
                            self.circular = 0
                        self.rotate_left = 0
                        degree = DEGREE
                    if action == 2:
                        self.rotate_left += 1
                        consec_rotate = self.rotate_left
                        if rotate_right>0:
                            self.circular += 1
                            reward -= self.circular*1000
                        else:
                            self.circular = 0
                        self.rotate_right = 0
                        degree = -DEGREE
                    threshold = int(config['recent_actions_history']/2.5)
                    self._rotate(config['rotate_degree'] * degree)
                    consecutive_rotate_threshold = 2
                    if consec_rotate > consecutive_rotate_threshold:
                        reward -= (consec_rotate - consecutive_rotate_threshold) * 1000
                    if self.recent_rotate_number > threshold:
                        reward -= (self.recent_rotate_number- threshold)*1000
            self.remaining_steps -= 1


            if action == 1 or action == 2:
                reward += self.calc_prospective_reward()

            if self.rotate_right != rotate_right:
                self.graphics.draw_number(self.coordinate, rotate_right, [SSL] * 3, clear=True)
                self.graphics.draw_number(self.coordinate, self.rotate_right, [SSL] * 3)

            if self.rotate_left != rotate_left:
                self.graphics.draw_number(self.coordinate2, rotate_left, [SSL] * 3, clear=True)
                self.graphics.draw_number(self.coordinate2, self.rotate_left, [SSL] * 3)

            if self.recent_rotate_number != recent_rotate:
                self.graphics.draw_number(self.coordinate3, recent_rotate, [SSL] * 3, clear=True)
                self.graphics.draw_number(self.coordinate3, self.recent_rotate_number, [SSL] * 3)

            self._done, tr = self._is_done()
            reward += tr
        self._draw_turtle(False)

        return self.remove_helper_channel(self.graphics.pixels), reward, self._done, {'coverage': self.coverage}

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
