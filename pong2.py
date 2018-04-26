import pygame
import numpy as np
import random
from math import *

white = [255, 255, 255]
black = [0, 0, 0]


def createSurf(w, h):
    return pygame.Surface((w, h))

def map(n, xStart, yStart, xTarget, yTarget):
    return xTarget + n / (yStart + xStart) * (yTarget - xTarget)


class env:

    def __init__(self, w, h):

        self.w = w
        self.h = h
        self.ball = Ball(w, h)
        self.player = [Player(w, h, True), Player(w, h, False)]
        self.surf = createSurf(w, h)
        self.terminal = False
        self.maxScore = 20
    
    def displayScore(self, side):
        
        pygame.init()
        font = pygame.font.SysFont(None, int(0.075 * self.h))

        s = {1: "main", 0: "copycat"}
        left_score = font.render("{} {}".format(s[side[0]], self.player[0].score), True, (255, 255, 255))
        self.surf.blit(left_score,(int(0.25*self.w), int(0.05 * self.h)))

        right_score = font.render("{} {}".format(s[side[1]], self.player[1].score), True, (255, 255, 255))
        self.surf.blit(right_score,(int(0.65*self.w), int(0.05 * self.h)))


    def step(self, action):

        if self.terminal:

            self.ball = Ball(self.w, self.h)
            self.player = [Player(self.w, self.h, True), Player(self.w, self.h, False)]
            self.terminal = False

        self.player[0].do(action[0])
        self.player[1].do(action[1])

        self.ball.update()
        self.reward = self.ball.edges([self.player[0].y, self.player[1].y])

        if 1 in self.reward:

            self.player[self.reward.index(1)].score += 1

        self.surf.fill(black)

        for player in self.player:
            player.edges()
            player.show(self.surf)

            if player.score == self.maxScore:

                self.terminal = True
            
        self.ball.show(self.surf)

        return pygame.surfarray.array3d(self.surf), self.reward, self.terminal


    def videoStep(self, action, side):

        self.score = [0, 0]

        if self.terminal:
            
            self.score = [0, 0]
            self.ball = Ball(self.w, self.h)
            self.player = [Player(self.w, self.h, True), Player(self.w, self.h, False)]
            self.terminal = False

        self.player[0].do(action[0])
        self.player[1].do(action[1])

        self.ball.update()
        self.reward = self.ball.edges([self.player[0].y, self.player[1].y])

        if 1 in self.reward:

            self.player[self.reward.index(1)].score += 1

        self.surf.fill(black)

        for player in self.player:
            player.edges()
            player.show(self.surf)

            if player.score == self.maxScore:

                self.terminal = True
            
        self.ball.show(self.surf)

        frame = pygame.surfarray.array3d(self.surf)

        self.displayScore(side)

        vFrame = pygame.surfarray.array3d(self.surf)

        return vFrame, frame, self.reward, self.terminal




class Ball:

    def __init__(self, w, h):

        self.x = w / 2
        self.y = h / 2
        self.h = h
        self.w = w
        self.acc = [0.04 * self.w, 0.04 * self.h]
        r = random.random()
        self.vel = [random.choice([-1, 1])*cos(r) * self.acc[0], sin(r) * self.acc[1]]
        self.radius = 0.02 * w
        self.reward = [0, 0]

    def update(self):

        self.x += self.vel[0]
        self.y += self.vel[1]

    def edges(self, pos):

        self.reward = [0, 0]

        if self.y > self.h - self.radius:
            self.y = self.h - self.radius
            self.vel[1] *= -1

        if self.y < self.radius:
            self.y = self.radius
            self.vel[1] *= -1

        if self.x < self.radius:
            self.x = self.radius
            self.vel[0] *= -1
            self.reward = [-1, 1]

        if self.x > self.w - self.radius:
            self.x = self.w - self.radius
            self.vel[0] *= -1
            self.reward = [1, -1]

        if self.x < 0.04 * self.w + self.radius and self.x > 0.04 * self.w - self.radius and self.y > pos[0] - self.radius and self.y < pos[0] + 0.2 * self.h + self.radius:
            self.x = 0.04 * self.w + self.radius
            a = map(self.y, pos[0] - self.radius, pos[0] + 0.2 * self.h + self.radius, -pi / 3, pi / 3)
            self.vel[0] = cos(a) * self.acc[0]
            self.vel[1] = sin(a) * self.acc[1]

        if self.x > self.w - (0.04 * self.w + self.radius) and self.x < self.w - (0.04 * self.w - self.radius) and self.y > pos[1] - self.radius and self.y < pos[1] + 0.2 * self.h + self.radius:
            self.x = self.w - (0.04 * self.w + self.radius)
            a = map(self.y, pos[1] - self.radius, pos[1] + 0.2 * self.h + self.radius, -pi / 3, pi / 3)
            self.vel[0] = -cos(a) * self.acc[0]
            self.vel[1] = sin(a) * self.acc[1]

        return self.reward

    def show(self, surf):

        pygame.draw.circle(surf, white, [int(self.x), int(self.y)], int(self.radius))


class Player:

    def __init__(self, w, h, isLeft):

        self.pos = [0, 0]
        self.h = h
        self.w = w
        self.y = h / 2
        self.score = 0
        self.vel = 0.04 * h
        self.width = 0.02 * w
        self.height = 0.2 * h
        self.pad = int(0.02 * w)

        if isLeft:
        	self.x = self.pad 
        else:
        	self.x = w - self.pad - self.width

    def edges(self):

        if self.y > self.h - self.height:
            self.y = self.h - self.height

        if self.y < 0:
            self.y = 0

    def do(self, action):

    	if action[0] == 1:
    		self.y += self.vel
    	if action[1] == 1:
    		self.y -= self.vel

    def show(self, surf):

        pygame.draw.rect(
            surf, white, [int(self.x), self.y, int(self.width), int(self.height)])
