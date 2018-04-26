import pong2 as game
import cv2
import numpy as np
import tensorflow as tf
import random
import os
import argparse
import pygame
import time
from train import *
from utils import *
from pygame.locals import*


def play(frame, logits, sess):
    pygame.init()   

    saver = tf.train.Saver()
    do_nothing = np.zeros((ACTIONS, ACTIONS))
    do_nothing[1] = 1

    sideArr = [np.ones((WIDTH, HEIGHT)), np.zeros((WIDTH, HEIGHT))]

    Game = game.env(210, 160)
    f_t, r_0, terminal = Game.step(do_nothing)

    f_t = resizeAndDiscolor(f_t)

    s_t = np.stack((sideArr[1], f_t, f_t, f_t), axis=2)
    
    screen = pygame.display.set_mode((Game.w, Game.h))

    t = 0
    sess.run(tf.global_variables_initializer())

    gamesPlayed = 0
    
    checkpoint = tf.train.get_checkpoint_state("model/{}".format(version))
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("model successfully loaded:", checkpoint.model_checkpoint_path)

    print("network copied into copycat")

    a = [0,0,1]


    while 1:

        time.sleep(1/24)

        for event in pygame.event.get():
            if event.type in [KEYUP,KEYDOWN]:
                print(event)
                if  event.key == K_UP:
                    if event.type == KEYDOWN:
                        a = [0,1,0]
                    else:
                        a = [0,0,1]
                elif event.key == K_DOWN:
                    if event.type == KEYDOWN:
                        a = [1,0,0]
                    else:
                        a = [0,0,1]
                else:
                    a = [0,0,1]
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        a_t = np.zeros([ACTIONS])

        output = logits.eval(feed_dict = {frame: [s_t]})[0]
        a_index = np.argmax(output)
        a_t[a_index] = 1

        action = [[],[]]
        action[0] = a
        action[1] = a_t
        
        f_t1, r_t, terminal = Game.step(action)

        surf = pygame.surfarray.make_surface(f_t1)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        f_t1 = resizeAndDiscolor(f_t1)
        f_t1 = np.stack((sideArr[1], f_t1), axis=2)
        f_t1 = np.reshape(f_t1, [WIDTH, HEIGHT, 2])
        s_t1 = np.append(f_t1, s_t[:, :, :2], axis=2)

        s_t = s_t1
           
def main():
    sess = tf.InteractiveSession()
    frame, logits = model()
    play(frame, logits, sess)

        
if __name__ == "__main__":
    main()
