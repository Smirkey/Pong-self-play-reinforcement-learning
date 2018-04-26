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


def infer(frame, frame_copy, logits, logits_copy, sess):

    saver = tf.train.Saver()
    do_nothing = np.zeros((ACTIONS, ACTIONS))
    do_nothing[1] = 1

    side = newSide()
    sideArr = [np.ones((WIDTH, HEIGHT)), np.zeros((WIDTH, HEIGHT))]

    Game = game.env(420, 320)
    vf, f_t, r_0, terminal = Game.videoStep(do_nothing, side)

    f_t = resizeAndDiscolor(f_t)

    s_t = np.stack((sideArr[side.index(1)], f_t, f_t, f_t), axis=2)
    s_t2 = np.stack((sideArr[side.index(0)], f_t, f_t, f_t), axis=2)
    

    screen = pygame.display.set_mode((Game.w, Game.h))

    t = 0
    sess.run(tf.global_variables_initializer())

    gamesPlayed = 0

    checkpoint = tf.train.get_checkpoint_state("model/{}".format(version))
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("model successfully loaded:", checkpoint.model_checkpoint_path)

    copy(sess)
    print("network copied into copycat")

    while 1:

        surf = pygame.surfarray.make_surface(vf)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        time.sleep(1/24)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        a_t = np.zeros([ACTIONS])
        a_t2 = np.zeros([ACTIONS])

        a_index = 2

        output = logits.eval(feed_dict = {frame: [s_t]})[0]
        output2 = logits_copy.eval(feed_dict={frame_copy: [s_t2]})[0]
        
        if t % FRAME_PER_ACTION == 0:
                a_index = np.argmax(output)
                a_t[a_index] = 1
                a_index = np.argmax(output2)
                a_t2[a_index] = 1
        else:
            a_t[a_index] = 1

        action = [[],[]]
        action[side.index(1)] = a_t
        action[side.index(0)] = a_t2

        vf, f_t1, r_t, terminal = Game.videoStep(action, side)

        f_t1 = resizeAndDiscolor(f_t1)

        f_t1_prime = np.stack((sideArr[side.index(0)], f_t1), axis=2)
        f_t1 = np.stack((sideArr[side.index(1)], f_t1), axis=2)
        

        f_t1 = np.reshape(f_t1, [WIDTH, HEIGHT, 2])
        f_t1_prime = np.reshape(f_t1_prime, [WIDTH, HEIGHT, 2])

        s_t1 = np.append(f_t1, s_t[:, :, :2], axis=2)
        s_t1_prime = np.append(f_t1_prime, s_t2[:, :, :2], axis=2)

        s_t = s_t1
        s_t2 = s_t1_prime
        t += 1
 
def main():
    sess = tf.InteractiveSession()
    frame, logits = model()
    frame_copy, logits_copy = copycat()
    infer(frame, frame_copy, logits, logits_copy, sess)
    
if __name__ == "__main__":
    main()
    
