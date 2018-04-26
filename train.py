
from collections import deque
from utils import *
import pong2 as game
import cv2
import numpy as np
import tensorflow as tf
import random
import os
import pygame

version = "test_0"
WIDTH = 80
HEIGHT = 80
NUMFRAMES = 3
ACTIONS = 3
GAMMA = 0.95
OBSERVE = 10000
EXPLORE = 2000000
FINAL_EPSILON = 0.001
START_EPSILON = 0.2
REPLAY_MEMORY = 50000
BATCH = 32
FRAME_PER_ACTION = 1


def model(reuse=False):
    f1, f2, f3 = 32, 64, 64

    frame = tf.placeholder(
        tf.float32, shape=[None, WIDTH, HEIGHT, NUMFRAMES + 1], name='frame')

    with tf.variable_scope('model') as scope:
        if reuse:
            scope.reuse_variables()

        conv1 = tf.layers.conv2d(frame, f1, kernel_size=[4, 4], strides=[
                                 4, 4], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv1')
        act1 = tf.nn.relu(conv1)

        conv2 = tf.layers.conv2d(act1, f2, kernel_size=[4, 4], strides=[
                                 2, 2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv2')
        act2 = tf.nn.relu(conv2)

        conv3 = tf.layers.conv2d(act2, f3, kernel_size=[4, 4], strides=[
                                 2, 2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv3')
        act3 = tf.nn.relu(conv3)

        fc1 = tf.reshape(act3, shape=[-1, 1600], name='fc1')

        w1 = tf.get_variable('w1', shape=[
                             1600, 512], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))

        b1 = tf.get_variable(
            'b1', shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        h_fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, w1), b1))

        w2 = tf.get_variable('w2', shape=[
                             512, ACTIONS], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))

        b2 = tf.get_variable(
            'b2', shape=[ACTIONS], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        logits = tf.matmul(h_fc1, w2) + b2

        return frame, logits

def update_epsilon(ϵ):

    ϵ -= (START_EPSILON - FINAL_EPSILON) / EXPLORE
    return ϵ


def train(frame, frame_copy, logits, logits_copy, sess):

    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    D = deque()

    saver = tf.train.Saver()
    readout_action = tf.reduce_sum(tf.multiply(logits, a), reduction_indices=1)
    loss = tf.reduce_mean(tf.square(y - readout_action))
    trainer = tf.train.AdamOptimizer(1e-6).minimize(loss)


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    do_nothing = np.zeros((ACTIONS,ACTIONS))
    do_nothing[1] = 1

    Game = game.env(210, 160)
    f_t, r_0, terminal = Game.step(do_nothing)
    f_t = resizeAndDiscolor(f_t)
    
    side = newSide()
    sideArr = [np.ones((WIDTH, HEIGHT)), np.zeros((WIDTH, HEIGHT))]

    s_t = np.stack((sideArr[side.index(1)], f_t, f_t, f_t), axis=2)
    s_t2 = np.stack((sideArr[side.index(0)], f_t, f_t, f_t), axis=2)

    gamesPlayed = 0

    epsilon = START_EPSILON

    t = 0
    k = 0

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    checkpoint = tf.train.get_checkpoint_state("model/{}".format(version))
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("model successfully loaded:", checkpoint.model_checkpoint_path)


    while 1:

        a_index = 2

        a_t = np.zeros([ACTIONS])
        a_t2 = np.zeros([ACTIONS])
        
        output = logits.eval(feed_dict={frame: [s_t]})[0]
        output2 = logits_copy.eval(feed_dict={frame_copy: [s_t2]})[0]

        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                a_index = random.randrange(ACTIONS)
                a_t[a_index] = 1
            else:
                a_index = np.argmax(output)
                a_t[a_index] = 1

            if random.random() <= epsilon:
                a_index = random.randrange(ACTIONS)
                a_t2[a_index] = 1
            else:
                a_index = np.argmax(output2)
                a_t2[a_index] = 1
        else:
            a_t[a_index] = 1
            a_t2[a_index] = 1

        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon = update_epsilon(epsilon)

        action = [[],[]]
        action[side.index(1)] = a_t
        action[side.index(0)] = a_t2

        f_t1, r_t, terminal = Game.step(action)

        if terminal:
            side = newSide()
            gamesPlayed += 1
            if gamesPlayed % 10 == 0:
                copy(sess)
                print("network copied")

        f_t1 = resizeAndDiscolor(f_t1)

        f_t1_prime = np.stack((sideArr[side.index(0)], f_t1), axis=2)
        f_t1 = np.stack((sideArr[side.index(1)], f_t1), axis=2)
        

        f_t1 = np.reshape(f_t1, [WIDTH, HEIGHT, 2])
        f_t1_prime = np.reshape(f_t1_prime, [WIDTH, HEIGHT, 2])

        s_t1 = np.append(f_t1, s_t[:, :, :2], axis=2)
        s_t1_prime = np.append(f_t1_prime, s_t2[:, :, :2], axis=2)

        D.append((s_t, a_t, r_t[side.index(1)], s_t1, terminal))

        if len(D) > REPLAY_MEMORY:
            D.popleft()

        if t > OBSERVE:

            minibatch = random.sample(D, BATCH)

            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]
            y_batch = []

            logits_j1_batch = logits.eval(feed_dict={frame: s_j1_batch})

            for i in range(len(minibatch)):
                terminal = minibatch[i][4]

                if terminal:
                    y_batch.append(r_batch[i])

                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(logits_j1_batch[i]))

            trainer.run(feed_dict={y: y_batch, a: a_batch, frame: s_j_batch})

        s_t = s_t1
        s_t2 = s_t1_prime

        if t % 50000 == 0 or t == 0:
            capture(frame, frame_copy, logits, logits_copy, epsilon, k)
            k+=1

        t += 1

        if t % 10000 == 0:
            if not os.path.exists('./model/' + version):
                os.makedirs('./model/' + version)
            saver.save(sess, './model/' + version + '/' + str(t))

        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        if t % 1000 == 0:
            print(output)
            print("Timestep: {}".format(t))
            print("State: {}, Epsilon: {}".format(state, epsilon))
            print("----------------")
  
        coord.request_stop()
        coord.join(threads)


def main():
    sess = tf.InteractiveSession()
    frame, logits = model()
    frame_copy, logits_copy = copycat()
    train(frame, frame_copy, logits, logits_copy, sess)


if __name__ == "__main__":
    main()

