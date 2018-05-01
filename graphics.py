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

DEGREE = 0.017453292519943295
EMPTY = 0.0
FULL = 255.0
SSL = 4 ## seven segment line
config = {
    "channel_numbers": 4,
    "maximum_steps": 75,
    "draw_channel": 0,
    "turtle_channel": 1,
    "target_channel": 2,
    "helper_channel": 3,
    "draw_reward": 20,
    "prospect_reward": 3,
    "draw_punish": 5,
    "time_punish": 5,
    "use_gpu_array": False,
    "target_line_width": 6,
    "turtle_line_width": 3,
    "end_state_reward": 4000,
    "recent_actions_history": 30,
    "rotate_degree": 90,
    "polygon_size": 4,
    "width": 10,
    "prospect_length":10
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
        flag = False
        data = EMPTY if inverse else FULL
        if channels == [config['draw_channel']]:
            if self.pixels[x, y, config['target_channel']] == FULL and self.pixels[x, y, config['draw_channel']] == FULL:
                self.pixels[x, y, config['helper_channel']]=FULL
            self.tlsp.append((x, y))

        for ch in channels:
            self.pixels[x, y, ch] = data



    def draw_polygon(self, points):
        start = points[0]
        self.target_p = 0
        for p in points[1:]:
            _, temp, _ = self.line(*start, *p, config['target_channel'], line_width=config['target_line_width'])
            self.target_p += temp
            start = p
        _, temp, _ = self.line(*p, *points[0], config['target_channel'], line_width=config['target_line_width'])
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
        _, temp, _ = self.line(x0, y0, x0 + width, y0, channels, line_width=config['target_line_width'])
        p+= temp
        _, temp, _ = self.line(x0, y0, x0, y0 + height, channels, line_width=config['target_line_width'])
        p+= temp
        _, temp, _ = self.line(x0 + width, y0, x0 + width, y0 + height, channels, line_width=config['target_line_width'])
        p+= temp
        _, temp, _ = self.line(x0, y0 + height, x0 + width, y0 + height, channels, line_width=config['target_line_width'])
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
                        return -0.2*config['draw_punish']
            else:
                return -1*config['draw_punish']
        except IndexError:
            return -1 * config['draw_punish']

    def is_in_points(self, x, y, container):
        return any((x, y) in l for l in container)

    def _plot(self, x, y, channels, inverse, line_width, calc_reward):

        reward, tr = 0, 0
        good_points_counter, bad_points_counter, tpc = 0, 0, 0
        line_width = int(line_width/2)

        for d in range(-line_width, line_width):
            for d2 in range(-line_width, line_width):
                try:
                    if channels==[config["draw_channel"]]:
                        if self.is_in_points(x+d, y+d2, [self.tlsp]):
                            continue
                        if self.nlsp and self.is_in_points(x+d, y+d2, self.lsp):
                            continue
                    temp = self.calc_point_reward(x+d, y+d2)
                    if calc_reward:
                        reward += temp
                        if temp>0:
                            good_points_counter += 1
                        elif self.pixels[x + d, y + d2, config['draw_channel']] == EMPTY:
                            bad_points_counter += 1
                    elif self.pixels[x+d, y+d2, config['target_channel']] == EMPTY:
                        good_points_counter += 1
                    self.plot(x + d, y + d2, channels, inverse)
                except IndexError as e:
                    # print(e)
                    #because of cursor index
                    continue
        return reward, good_points_counter, bad_points_counter

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
                line_width = int(config['target_line_width']/3)
            if(a==0):
                _, temp, _ = self.line(origin[0], origin[1], origin[0]+SSL[0], origin[1], chan, clear, line_width)
            elif(a==1):
                _, temp, _ = self.line(origin[0], origin[1], origin[0], origin[1]+SSL[1], chan, clear, line_width)
            elif(a==2):
                _, temp, _ = self.line(origin[0]+SSL[0], origin[1], origin[0]+SSL[0], origin[1]+SSL[1], chan, clear, line_width)
            elif(a==3):
                _, temp, _ = self.line(origin[0], origin[1]+SSL[1], origin[0]+SSL[0], origin[1]+SSL[1], chan, clear, line_width)
            elif(a==4):
                _, temp, _ = self.line(origin[0], origin[1]+SSL[1], origin[0], origin[1]+SSL[1]+SSL[2], chan, clear, line_width)
            elif(a==5):
                _, temp, _ = self.line(origin[0]+SSL[0], origin[1]+SSL[1], origin[0]+SSL[0], origin[1]+SSL[1]+SSL[2], chan, clear, line_width)
            elif(a==6):
                _, temp, _ = self.line(origin[0], origin[1]+SSL[1]+SSL[2], origin[0]+SSL[0], origin[1]+SSL[1]+SSL[2], chan, clear, line_width)
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
                tr, tgpc, tbpc = self._plot(x, y, channels, inverse, line_width, calc_reward)
                reward += tr
                gpc += tgpc
                bpc += tbpc
                error += deltaerr
                while error >= 0.5:
                    y += sgn
                    tr, tgpc, tbpc  = self._plot(x, y, channels, inverse, line_width, calc_reward)
                    reward += tr
                    gpc += tgpc
                    bpc += tbpc
                    error -= 1.
        else:
            if y1 < y0:
                y1, y0 = y0, y1
            for y in range(y0, y1):
                tr, tgpc, tbpc  = self._plot(x0, y, channels, inverse, line_width, calc_reward)
                reward += tr
                gpc += tgpc
                bpc += tbpc
        if (channels == [config["draw_channel"]]):
            self.last_line = [(x0, y0), (x1, y1)]

        return reward, gpc, bpc


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
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width, 3))
        self.display = None
        self.coordinate = [int(self.width*5/16), int(self.width/18)]
        self.coordinate2 = [int(self.width*1/16), int(self.width/18)]

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


    def reset(self):
        self.graphics.reset()
        self.angle = math.pi/2.
        self.rotate_step = 0
        self.recent_actions = []
        self.recent_rotate_number = 0
        self.pen = True
        self.pen_repeat = False
        self.remaining_steps = self.maximum_steps
        x = random.sample(range(10,self.screen_width-20), 1)[0]
        y = random.sample(range(15, self.screen_height-45), 1)[0]
        origin = [x, y]
        size = np.random.choice(range(3,5), 3)*5
        target_num = random.sample(range(2,6), 1)
        self.graphics.draw_number(origin, target_num[0], size, chan=config['target_channel'])
        self.graphics.draw_number(self.coordinate, self.rotate_step, [SSL] * 3)
        self.graphics.draw_number(self.coordinate2, self.recent_rotate_number, [SSL] * 3)

        #random polygon
        # polygon_points = self._select_polygon_points(config["polygon_size"])
        # self.graphics.draw_polygon(polygon_points)

        #random rectangle
        # point = self.np_random.randint(10, self.screen_width/2, 2)
        # self.graphics.rect(*point, size[0], size[1])

        self.tx = origin[0]
        self.ty = origin[1]
        self._draw_turtle()
        self._done = False


        # num = self.np_random.randint(3, 8)
        # self._draw_polygon(num)
        return np.uint8(self.graphics.pixels)

    def _forward(self, distance):
        self.graphics.lsp.append(self.graphics.tlsp)
        self.graphics.tlsp = []
        reward, p = 0, 0
        cx = int(distance * math.cos(self.angle))
        cy = int(distance * math.sin(self.angle))
        self._draw_turtle(True)
        if (self.tx + cx > (self.screen_width+5) or self.ty + cy > (self.screen_height+5) or self.ty + cy < SSL-5 or self.tx + cx < -5):
            self._rotate(math.pi, False)
            cx = int(distance * math.cos(self.angle))
            cy = int(distance * math.sin(self.angle))
        if self.pen:
            reward, gp, bp = self.graphics.line(self.tx, self.ty, self.tx + cx, self.ty + cy,
                                        channels=[config['draw_channel']], line_width=config['target_line_width'],
                                        calc_reward=True)

        self.tx += cx
        self.ty += cy
        self._draw_turtle(False)
        self.graphics.draw_p += gp
        self.graphics.draw_wrong_p += bp
        return reward

    def _rotate(self, angle, draw=True):
        if draw:
            self._draw_turtle(True)
        self.angle += angle
        if self.angle < 0.:
            self.angle += 2 * math.pi
        elif self.angle > 2 * math.pi:
            self.angle -= 2 * math.pi
        if draw:
            self._draw_turtle(False)

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

    def render(self, mode='human', close=False):
        if(self.display==None):
            pygame.init()
            display = pygame.display.set_mode((self.width, self.height))
        # time.sleep(0.01)
        temp = self.graphics.pixels[:,:,1] + self.graphics.pixels[:,:,3]
        data = np.uint8(np.stack((self.graphics.pixels[:,:,0], temp, self.graphics.pixels[:,:,2]), axis = 2))
        surf = pygame.surfarray.make_surface(data)
        display.blit(surf, (0, 0))
        pygame.display.update()
        return


    def _is_done(self):
        coverage = float(self.graphics.draw_p)/float(self.graphics.target_p)
        if coverage > 0.99:
            print("coverage", coverage)
            end_reward_coeff = (self.graphics.draw_p) / (self.graphics.draw_wrong_p*1.5 + self.graphics.draw_p)
            return True, config['end_state_reward'] * end_reward_coeff
        if self.remaining_steps == 0:
            print("coverage", coverage)
            return True, 0
        return False, 0

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

#change logger
# make lines thicker

    def calc_prepend_points(self, p):
        width = config['width']
        p1y = math.sin(self.angle + math.pi / 2) * width + p[1]
        p1x = math.cos(self.angle + math.pi / 2) * width + p[0]
        p2y = math.sin(self.angle - math.pi / 2) * width + p[1]
        p2x = math.sin(self.angle - math.pi / 2) * width + p[0]
        return ((int(p1x), int(p1y)), (int(p2x), int(p2y)))

    def calc_prospective_reward(self):
        fps = self.calc_prepend_points((self.tx, self.ty))
        prospect = config['prospect_length']
        npoint = (math.cos(self.angle)*prospect+self.tx, math.sin(self.angle)*prospect+self.ty)
        sps = self.calc_prepend_points(npoint)

        #TODO does not work properly for fractional degrees
        xs = [fps[0][0], fps[1][0], sps[0][0], sps[1][0]]
        ys = [fps[0][1], fps[1][1], sps[0][1], sps[1][1]]
        minx = min(xs)
        maxx = max(xs)
        miny = min(ys)
        maxy = max(ys)
        print(xs, ys)
        rectangle_view = np.array(self.graphics.pixels[minx:maxx, miny:maxy,:])
        target_points = np.logical_and(rectangle_view[:,:,2]==FULL, rectangle_view[:,:,0]==EMPTY)
        print(target_points)
        total_points = np.prod(rectangle_view.shape)
        return (np.count_nonzero(target_points)/total_points)*config['prospect_reward']


    def step(self, action):
        self.graphics.draw_number(self.coordinate, self.rotate_step, [SSL]*3, clear = True)
        self.graphics.draw_number(self.coordinate2, self.recent_rotate_number, [SSL]*3, clear = True)
        self.recent_actions.append(action)
        if(len(self.recent_actions)>config['recent_actions_history']):
            del self.recent_actions[0]

        self.recent_rotate_number = reduce(lambda x,y: x+1 if y==1 else x, [0, *self.recent_actions])
        if self._done == True:
            return np.uint8(self.graphics.pixels), 0 , True, {}
        self.remaining_steps -= 1
        reward = -config['time_punish']
        if action == 0:
            reward += self._forward(5)
            self.rotate_step=0
        elif action == 1:
            self.rotate_step += 1
            threshold = config['recent_actions_history']/2
            self._rotate(config['rotate_degree'] * DEGREE)
            consecutive_rotate_threshold = (360/config['rotate_degree'])-1
            if self.rotate_step > consecutive_rotate_threshold:
                reward -= (self.rotate_step - consecutive_rotate_threshold) * 500
            if self.recent_rotate_number > threshold:
                reward -= (self.recent_rotate_number- threshold)*500
            # if self.rotate_step > (180/config['rotate_degree']-1):
            #     print("shod")
            #     self.graphics.lsp = [

        elif action == 2:
            pass
        print("prospect reward", self.calc_prospective_reward())
        self.graphics.draw_number(self.coordinate, self.rotate_step, [SSL]*3)
        self.graphics.draw_number(self.coordinate2, self.recent_rotate_number, [SSL]*3)

        coverage = float(self.graphics.draw_p)/float(self.graphics.target_p)

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
        self._done, tr = self._is_done()
        reward+=tr

        return np.uint8(self.graphics.pixels), reward, self._done, {'coverage': coverage}

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
        print(i, end-start, rt)
    print(total/10.0)


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
