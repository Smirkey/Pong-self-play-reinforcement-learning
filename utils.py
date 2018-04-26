import cv2
import numpy as np
import tensorflow as tf
import random
import os
import argparse
import pong2 as game

WIDTH = 80
HEIGHT = 80
NUMFRAMES = 3
ACTIONS = 3
version = "test_0"
FRAME_PER_ACTION = 1


def copycat(reuse=False):
    f1, f2, f3 = 32, 64, 64

    frame_copy = tf.placeholder(
        tf.float32, shape=[None, WIDTH, HEIGHT, NUMFRAMES + 1], name='frame_copy')

    with tf.variable_scope('copycat') as scope:
        if reuse:
            scope.reuse_variables()

        conv1_copy = tf.layers.conv2d(frame_copy, f1, kernel_size=[4, 4], strides=[
                                 4, 4], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv1_copy')
        act1_copy = tf.nn.relu(conv1_copy)

        conv2_copy = tf.layers.conv2d(act1_copy, f2, kernel_size=[4, 4], strides=[
                                 2, 2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv2_copy')
        act2_copy = tf.nn.relu(conv2_copy)

        conv3_copy = tf.layers.conv2d(act2_copy, f3, kernel_size=[4, 4], strides=[
                                 2, 2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv3_copy')
        act3_copy = tf.nn.relu(conv3_copy)

        fc1_copy = tf.reshape(act3_copy, shape=[-1, 1600], name='fc1_copy')

        w1_copy = tf.get_variable('w1_copy', shape=[
                             1600, 512], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))

        b1_copy = tf.get_variable(
            'b1_copy', shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        h_fc1_copy = tf.nn.relu(tf.add(tf.matmul(fc1_copy, w1_copy), b1_copy))

        w2_copy = tf.get_variable('w2_copy', shape=[
                             512, ACTIONS], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))

        b2_copy = tf.get_variable(
            'b2_copy', shape=[ACTIONS], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        logits_copy = tf.matmul(h_fc1_copy, w2_copy) + b2_copy
        return frame_copy, logits_copy


def copy(sess):

    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')[:10]

    for variable in variables:
        sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='copycat')[variables.index(variable)].assign(variable))

    del variables

def resizeAndDiscolor(img):

    img = cv2.cvtColor(cv2.resize(img, (WIDTH, HEIGHT)), cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    return img


def newSide():

    side = [0,0]
    side[random.choice((0,1))] = 1
    return side

def makeVideo(frames, k):
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=False, default='./games/{}/{}.avi'.format(version, k), help="output video file")
    args = vars(ap.parse_args())
    output = args['output']
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
    height, width, channels = frames[0].shape
    print(width, height, channels)
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

    if not os.path.exists('./games/' + version):
                os.makedirs('./games/' + version)
    
    for frame in frames:
        out.write(frame)

    out.release()
    print("video out")

def capture(frame, frame_copy, logits, logits_copy, e, k):

    frames = []

    do_nothing = np.zeros([ACTIONS, ACTIONS])
    testEnv = game.env(210, 160)

    side = newSide()
    sideArr = [np.ones((WIDTH, HEIGHT)), np.zeros((WIDTH, HEIGHT))]

    vf, f, r, t = testEnv.videoStep(do_nothing, side)
    frames.append(cv2.resize(vf, (500, 500)))
    f = resizeAndDiscolor(f)

    s = np.stack((sideArr[side.index(1)], f, f, f), axis=2)
    s2 = np.stack((sideArr[side.index(0)], f, f, f), axis=2)

    a_index = 2

    while not t:

        a = np.zeros([ACTIONS])
        a2 = np.zeros([ACTIONS])
        
        o = logits.eval(feed_dict={frame: [s]})[0]
        o2 = logits_copy.eval(feed_dict={frame_copy: [s2]})[0]

        if t % FRAME_PER_ACTION == 0:
            if random.random() <= e:
                a_index = random.randrange(ACTIONS)
                a[a_index] = 1
            else:
                a_index = np.argmax(o)
                a[a_index] = 1

            if random.random() <= e:
                a_index = random.randrange(ACTIONS)
                a2[a_index] = 1
            else:
                a_index = np.argmax(o2)
                a2[a_index] = 1
        else:
            a[a_index] = 1
            a2[a_index] = 1

        action = [[],[]]
        action[side.index(1)] = a
        action[side.index(0)] = a2

        vf, f1, r, t = testEnv.videoStep(action, side)

        vf = cv2.flip(vf, -1)
        vf = cv2.flip(vf, 0)
        vf = cv2.resize(vf, (500, 500))

        frames.append(vf)

        f1 = resizeAndDiscolor(f1)

        f1p = np.stack((sideArr[side.index(0)], f1), axis=2)
        f1 = np.stack((sideArr[side.index(1)], f1), axis=2)

        f1 = np.reshape(f1, [WIDTH, HEIGHT, 2])
        f1p = np.reshape(f1p, [WIDTH, HEIGHT, 2])

        s1 = np.append(f1, s[:, :, :2], axis=2)
        s1p = np.append(f1p, s2[:, :, :2], axis=2)

        s = s1
        s2 = s1p

    makeVideo(frames, k)

    del vf, f1, f1p, s1, s1p, s, s2, testEnv


