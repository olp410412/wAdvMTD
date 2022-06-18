"""

在MNIST数据集上进行实验的 模型 统合版
"""
import datetime
import os
import sys
import time
import pickle5 as pickle

import numpy as np
import math
import matplotlib
# matplotlib.use('Agg')           # noqa: E402
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy.signal import convolve2d
import tensorflow as tf


# 模型定义部分

def my_model_cnn(x, logits=False, training=False):
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=128, kernel_size=[5, 5],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    # with tf.variable_scope('conv2'):
    #    z = tf.layers.conv2d(z, filters=128, kernel_size=[3, 3],
    #                         padding='same', activation=tf.nn.relu)
    #    z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
        z = tf.layers.dropout(z, rate=0.25, training=training)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


def my_model_test(x, logits=False, training=False):
    with tf.variable_scope('conv11'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv12'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('flatten11'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])

    with tf.variable_scope('mlp11'):
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
        z = tf.layers.dropout(z, rate=0.25, training=training)

    logits_ = tf.layers.dense(z, units=10, name='logits1')
    y = tf.nn.softmax(logits_, name='ybar1')

    if logits:
        return y, logits_
    return y


def my_model_test1(x, logits=False, training=False):
    with tf.variable_scope('conv21'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv22'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('flatte21'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])

    with tf.variable_scope('mlp21'):
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
        z = tf.layers.dropout(z, rate=0.25, training=training)

    logits_ = tf.layers.dense(z, units=10, name='logits2')
    y = tf.nn.softmax(logits_, name='ybar2')

    if logits:
        return y, logits_
    return y


def my_model_VGG():
    y = 0
    return y


def my_model_ResNet50():
    y = 0
    return y


class Dummy:
    pass


# 对抗模型定义部分
# 有两个candidates是相似的，评判结果时注意这一点，以免误判


def fgmt(model, model1, model2, x, y=None, eps=0.01, epochs=1, sign=True, clip_min=0.,
         clip_max=1.):
    """
    Fast gradient method with target

    See https://arxiv.org/pdf/1607.02533.pdf.  This method is different from
    FGM that instead of decreasing the probability for the correct label, it
    increases the probability for the desired label.

    :param model3:
    :param model2:
    :param model1:
    :param model: A model that returns the output as well as logits.
    :param x: The input placeholder.
    :param y: The desired target label, set to the least-likely class if None.
    :param eps: The noise scale factor.
    :param epochs: Maximum epoch to run.
    :param sign: Use gradient sign if True, otherwise gradient values.
    :param clip_min: Minimum value in output.
    :param clip_max: Maximum value in output.
    """
    # tf.identity 在这里的作用是赋值。若直接让xadv=x 属于引用传值，改变xadv不会生效，因为x没有改变。由于最终要由x生成新的xadv，所以xadv的值需要改变，所以使用tf.identity
    xadv = tf.identity(x)
    xadv1 = tf.identity(x)
    xadv2 = tf.identity(x)
    # epochs 在加噪过程中控制的是 Epsilon 叠加的次数，n epochs就叠加n次，然后再通过clip约束在范围内
    ybar = model(xadv)
    ydim = ybar.get_shape().as_list()[1]
    n = tf.shape(ybar)[0]

    if y is None:
        indices = tf.argmin(ybar, axis=1)
    else:
        indices = tf.cond(tf.equal(0, tf.rank(y)),
                          lambda: tf.zeros([n], dtype=tf.int32) + y,
                          lambda: tf.zeros([n], dtype=tf.int32))

    target = tf.cond(
        tf.equal(ydim, 1),
        lambda: 1 - ybar,
        lambda: tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0))

    if 1 == ydim:
        loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
    else:
        loss_fn = tf.nn.softmax_cross_entropy_with_logits

    if sign:
        noise_fn = tf.sign
    else:
        noise_fn = tf.identity

    eps = -tf.abs(eps)

    # 负号 是让他们（x） 沿损失函数梯度的相反方向 增长。即：与当前梯度下降 完全相反方向上，走Epsilon个单位。达到提高损失函数的目的

    def _cond(adv, adv1, adv2, i):
        return tf.less(i, epochs)

    def _body(xadv, xadv1, xadv2, i):
        # 使用FGSM 对第一个异构体产生基于梯度的对抗样本：xadv
        ## print not affect here print('deploying FGSM on model1\n')
        ybar, logits = model(xadv, logits=True)
        loss = loss_fn(labels=target, logits=logits)
        dy_dx, = tf.gradients(loss, xadv)
        # STOP GRAD 指的是不让他们参与到目标模型的更新中，否则就相当于对模型进行了继续训练
        xadv = tf.stop_gradient(xadv + eps * noise_fn(dy_dx))
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)

        # 使用FGSM 对第二个异构体产生基于梯度的对抗样本：xadv1
        ## print('deploying FGSM on model2\n')
        ybar1, logits1 = model1(xadv1, logits=True)
        loss1 = loss_fn(labels=target, logits=logits1)
        dy_dx1, = tf.gradients(loss1, xadv1)
        xadv1 = tf.stop_gradient(xadv1 + eps * noise_fn(dy_dx1))
        xadv1 = tf.clip_by_value(xadv1, clip_min, clip_max)

        # 使用FGSM 对第三个异构体产生基于梯度的对抗样本：xadv2
        ## print('deploying FGSM on model2\n')
        ybar2, logits2 = model2(xadv2, logits=True)
        loss2 = loss_fn(labels=target, logits=logits2)
        dy_dx2, = tf.gradients(loss2, xadv2)
        xadv2 = tf.stop_gradient(xadv2 + eps * noise_fn(dy_dx2))
        xadv2 = tf.clip_by_value(xadv2, clip_min, clip_max)

        print(loss)

        return xadv, xadv1, xadv2, i + 1

    xadv, xadv1, xadv2, _ = tf.while_loop(_cond, _body, (xadv, xadv1, xadv2, 0), back_prop=False,
                                          name='fast_gradient_target')
    # 在这里，考虑如何对多各异构体同时发起白盒攻击
    # 分别求三次导？然后生成三个adv
    # 三个adv要么平均，要么从中选一个，然后放在源样本上
    return xadv, xadv1, xadv2


def make_fgmt(sess, env, X_data, epochs=1, eps=0.01, batch_size=128):
    """
    Generate FGSM by running env.x_fgsm.
    """
    print('\nMaking adversarials via FGSM')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        # 参数实际起作用的地方
        # 求：env.x_fgmt的值，用到的数据在 feeddict中。然后调用env.x_fgmt赋值的语句
        # muted break point here
        adv, adv1, adv2 = sess.run(env.x_fgmt,
                                   feed_dict={env.x: X_data[start:end], env.fgsm_eps: eps, env.fgsm_epochs: epochs})
        # 衡量重加噪有效性的
        # 第一个模型
        X_adv[start:end] = adv
        # 衡量多数表决有效性
        # X_adv[start:end] = (adv + adv1 + adv2) / 3
        # print(adv, adv1, adv2)
        # 如何组合噪声 不同的组合方式会使图像相似度发生变化 这样的组合方式可以绘成一个曲线 alpha * adv + beta * adv2. 加噪效果随alpha 和 beta 的变化情况
        print()
    return X_adv


# 函数部分


def evaluate(sess, env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    loss, acc = 0, 0
    loss1, acc1 = 0, 0
    loss2, acc2 = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start

        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})

        batch_loss1, batch_acc1 = sess.run(
            [env.loss1, env.acc1],
            feed_dict={env.x: X_data[start:end],
                       env.y1: y_data[start:end]})

        batch_loss2, batch_acc2 = sess.run(
            [env.loss2, env.acc2],
            feed_dict={env.x: X_data[start:end],
                       env.y2: y_data[start:end]})

        loss += batch_loss * cnt
        acc += batch_acc * cnt
        loss1 += batch_loss1 * cnt
        acc1 += batch_acc1 * cnt
        loss2 += batch_loss2 * cnt
        acc2 += batch_acc2 * cnt

    loss /= n_sample
    acc /= n_sample
    loss1 /= n_sample
    acc1 /= n_sample
    loss2 /= n_sample
    acc2 /= n_sample

    print(
        ' loss: {0:.4f} acc: {1:.4f} loss1: {2:.4f} acc1: {3:.4f} loss2: {4:.4f} acc2: {5:.4f}'.format(loss, acc, loss1,
                                                                                                       acc1, loss2,
                                                                                                       acc2))

    # 实际上，没有用到返回值，loss计算出来后是直接在tf的session中进行更新的
    return loss, acc, loss1, acc1, loss2, acc2


def train(sess, env, X_data, y_data, X_valid=None, y_valid=None, epochs=5,
          load=False, shuffle=True, batch_size=256, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    # if load:
    #    if not hasattr(env, 'saver'):
    #        return print('\nError: cannot find saver op')
    #    print('\nLoading saved model')
    #    return env.saver.restore(sess, 'model/{}'.format(name))

    print('\nTrain model')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))

        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            ## print('updating parameters of model1')
            sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                              env.y: y_data[start:end],
                                              env.training: True})
            ## print('updating parameters of model2')
            sess.run(env.train_op1, feed_dict={env.x: X_data[start:end],
                                               env.y1: y_data[start:end],
                                               env.training1: True})

            sess.run(env.train_op2, feed_dict={env.x: X_data[start:end],
                                               env.y2: y_data[start:end],
                                               env.training2: True})
        if X_valid is not None:
            evaluate(sess, env, X_valid, y_valid)

    if hasattr(env, 'saver'):
        print('\n Saving model')
        os.makedirs('../../model', exist_ok=True)
        env.saver.save(sess, 'model/{}'.format(name))


def predict(sess, env, X_data, batch_size=256):
    """
    Do inference by running env.ybar.
    """
    print('\nPredicting')
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    yval = np.empty((n_sample, n_classes))
    yval1 = np.empty((n_sample, n_classes))
    yval2 = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        ## print('\t using model1 to pred')
        y_batch = sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
        ## print('\t using model2 to pred')
        y_batch1 = sess.run(env.ybar1, feed_dict={env.x: X_data[start:end]})
        yval1[start:end] = y_batch1

        y_batch2 = sess.run(env.ybar2, feed_dict={env.x: X_data[start:end]})
        yval2[start:end] = y_batch2
    print()
    return yval, yval1, yval2


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))


def compute_mse(im1, im2):
    mse = np.mean((im1 - im2) ** 2)
    return mse


def compute_psnr(im1, im2):
    mse = compute_mse(im1, im2)
    if 0 == mse:
        return 100
    PIXEL_MAXIMUM = 255.01
    return 20 * math.log10(PIXEL_MAXIMUM / math.sqrt(mse))


# 训练逻辑部分（最好能与未来的调度、异构训练对接）


def setting_up_env(env, my_number=2):
    # 给 env 赋值
    ## 目前只支持最多5个异构体
    with tf.variable_scope('model'):
        env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                               name='x')
        env.y = tf.placeholder(tf.float32, (None, n_classes), name='y')

        env.y1 = tf.placeholder(tf.float32, (None, n_classes), name='y1')
        env.y2 = tf.placeholder(tf.float32, (None, n_classes), name='y2')

        env.training = tf.placeholder_with_default(False, (), name='mode')
        env.training1 = tf.placeholder_with_default(False, (), name='mode1')
        env.training2 = tf.placeholder_with_default(False, (), name='mode2')

        env.ybar, logits = my_model_cnn(env.x, logits=True, training=env.training)
        env.ybar1, logits1 = my_model_test(env.x, logits=True, training=env.training1)
        env.ybar2, logits2 = my_model_test1(env.x, logits=True, training=env.training2)

        with tf.variable_scope('acc'):
            count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
            env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

            count1 = tf.equal(tf.argmax(env.y1, axis=1), tf.argmax(env.ybar1, axis=1))
            env.acc1 = tf.reduce_mean(tf.cast(count1, tf.float32), name='acc1')
            count2 = tf.equal(tf.argmax(env.y2, axis=1), tf.argmax(env.ybar2, axis=1))
            env.acc2 = tf.reduce_mean(tf.cast(count2, tf.float32), name='acc2')

        with tf.variable_scope('loss'):
            xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                           logits=logits)
            env.loss = tf.reduce_mean(xent, name='loss')

            xent1 = tf.nn.softmax_cross_entropy_with_logits(labels=env.y1,
                                                            logits=logits1)
            env.loss1 = tf.reduce_mean(xent1, name='loss1')

            xent2 = tf.nn.softmax_cross_entropy_with_logits(labels=env.y2,
                                                            logits=logits2)
            env.loss2 = tf.reduce_mean(xent2, name='loss2')

        with tf.variable_scope('train_op'):
            optimizer = tf.train.AdamOptimizer()
            env.train_op = optimizer.minimize(env.loss)

            optimizer1 = tf.train.AdamOptimizer()
            env.train_op1 = optimizer1.minimize(env.loss1)
            optimizer2 = tf.train.AdamOptimizer()
            env.train_op2 = optimizer2.minimize(env.loss2)

        env.saver = tf.train.Saver()

    # 自然要给 env.fgsm等 赋值，所以在赋值过程中，调用了fgmt函数
    with tf.variable_scope('model', reuse=True):
        env.fgsm_eps = tf.placeholder(tf.float32, (), name='fgsm_eps')
        env.fgsm_epochs = tf.placeholder(tf.int32, (), name='fgsm_epochs')
        # fgsm 产生对于 my_model_cnn 的对抗样本 （目前可以分别产生）（最好是在训练中产生，即求导阶段）
        # 还可以在env中追加其他算法，然后加if条件句，根据flag 决定是否更新该参数 方法类似下面一句
        env.x_fgmt = fgmt(my_model_cnn, my_model_test, my_model_test1, env.x, epochs=env.fgsm_epochs, eps=env.fgsm_eps)

    print('\nInitializing graph')
    return 0


# 目标 训练一个adv
# 投票表决的优势在于 构造一个能让三个异构模型同样的偏移 难于 让三个异构体都产生偏移


def my_re_noise(X_adv, X_test, mode=1, pos_num=1, stride=0.02, dire=0):
    eta = X_adv - X_test
    if pos_num < 1:
        print("pos_num_eor")
        return X_adv
    # 生成噪声
    if 1 == mode:
        # 改变一个 或 在多个位置各自改变一个
        for my_i in range(0, pos_num):
            re_noise_pixel = np.random.random()
            # 这里应该命名为行、列的，用x y表述，顺序这里是反的，先不管
            pos_x = np.random.random_integers(0, 27)
            pos_y = np.random.random_integers(0, 27)
            if re_noise_pixel >= 0.5 and 0 == dire:
                eta[:, pos_x, pos_y, 0] += stride
            elif re_noise_pixel < 0.5 and 0 == dire:
                eta[:, pos_x, pos_y, 0] -= stride
            else:
                eta[:, pos_x, pos_y, 0] -= stride
            # print('pos_x={0}, pos_y={1}, re_noise_pixel={2}'.format(pos_x, pos_y, re_noise_pixel))

    elif 2 == mode:
        # 改变一行或一列 或 改变多行
        for my_i in range(0, pos_num):
            re_noise_vec = np.random.random()
            pos = np.random.random_integers(0, 27)
            if re_noise_vec >= 0.5:
                eta[:, pos, :, 0] += stride
            elif re_noise_vec < 0.5:
                eta[:, pos, :, 0] -= stride

    elif 3 == mode:
        # 改变整个图像
        for my_i in range(0, pos_num):
            re_noise_vec = np.random.random()
            if re_noise_vec >= 0.5:
                eta[:, :, :, 0] += stride
    if 4 == mode:
        for my_i in range(0, pos_num):
            re_noise_pixel = np.random.random()
            # 这里应该命名为行、列的，用x y表述，顺序这里是反的，先不管
            pos_x = np.random.random_integers(0, 27)
            pos_y = np.random.random_integers(0, 27)
            if re_noise_pixel >= 0.5 and 0 == dire:
                eta[my_i, pos_x, pos_y, 0] += stride
            elif re_noise_pixel < 0.5 and 0 == dire:
                eta[my_i, pos_x, pos_y, 0] -= stride
            else:
                eta[my_i, pos_x, pos_y, 0] -= stride
            # print('pos_x={0}, pos_y={1}, re_noise_pixel={2}'.format(pos_x, pos_y, re_noise_pixel))
    else:
        return X_adv

    # 决定加噪点
    X_adv = eta + X_test
    np.clip(X_adv, 0, 1)
    return X_adv


def my_HD(array1, array2):
    hd = 0
    list1 = array1.tolist()
    list2 = array2.tolist()
    length1 = len(list1)
    length2 = len(list2)
    if length1 != length2:
        print("hd eor")
        return 0
    for my_i in range(0, length1):
        if list1[my_i] != list2[my_i]:
            hd += 1
    return hd


def my_vote_3(array1, array2, array3):
    # 有绝对多数就按绝对多数
    # 相对多数按参数讨论
    # 偏差性过大 直接分类为垃圾样本（检测出是对抗样本，归类为垃圾，这个也能作为contribution来写）

    return 0


def my_single_pixel(X_adv, X_test, my_batch_size=10000):
    tmp_adv = np.zeros((200000, 28, 28, 1))
    tmp_test = np.zeros((200000, 28, 28, 1))
    tmp_test_2 = np.zeros((200000))

    for my_i in range(100, 200):
        y1, y11, y111 = predict(sess, my_env_1, X_test)
        z1 = np.argmax(y1, axis=1)

        for my_j in range(0, 200000):
            tmp_adv[my_j, :, :, :] = X_adv[my_i, :, :, :]
            tmp_test[my_j, :, :, :] = X_test[my_i, :, :, :]
            tmp_test_2[my_j] = z1[my_i]

        y2, y22, y222 = predict(sess, my_env_1, tmp_adv)
        z2 = np.argmax(y2, axis=1)
        a1 = my_HD(tmp_test_2, z2)

        if 0 == a1:
            print("Necessity = 0, continue")
            continue

        for my_j in range(1, 2):
            for my_k in range(0, 13):
                X_adv2 = my_re_noise(tmp_adv, tmp_test, mode=4, pos_num=200000, stride=0.02 * my_k)
                delta = X_adv2 - tmp_adv
                if my_i >= 1000:
                    delta = np.squeeze(delta)
                    debug1 = np.squeeze(X_adv2)
                    debug2 = np.squeeze(tmp_adv)
                y3, y33, y333 = predict(sess, my_env_1, X_adv2)
                z3 = np.argmax(y3, axis=1)
                a2 = my_HD(z2, z3)

                print(
                    'rateA = {0}, stride = {1}, distance = {2}, 扰动率 = {3}, 扰动强度= {4}, necessity = {5}'.format(my_j,
                                                                                                              my_k, a2,
                                                                                                              my_j / 784,
                                                                                                              my_k / 12,
                                                                                                              a1))
                # my_diverge[tmp, my_k] = a1

    return 0


def my_noise_shuffle(X_adv, X_test, sample_size=200000, my_epochs=10):
    tmp1 = X_adv - X_test
    tmp2 = tmp1[0, :, :, :]
    tmp4 = np.squeeze(tmp2)

    for my_i in range(0, my_epochs):

        my_batch = np.empty((sample_size, 28, 28, 1))
        flag2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for index in range(0, sample_size):
            np.random.shuffle(tmp4)
            # np.random.shuffle(tmp4.T)
            tmp5 = tmp4 + np.squeeze(X_test[0, :, :, :])
            target = np.reshape(tmp5, (-1, 28, 28, 1))
            my_batch[index, :, :, :] = target

        my_y, _, _ = predict(sess, my_env_1, my_batch)
        my_z = np.argmax(my_y, axis=1)

        for index in range(0, sample_size):
            index1 = my_z[index]
            flag2[index1] += 1
        print(flag2)
        # tmp3 = np.squeeze(tmp2)

    return 0


def my_make_re_noise(X_adv, X_test):
    my_diverge = np.ndarray((13, 13))

    for my_j in range(0, 2):
        # for my_j in range(0, 520, 40):
        tmp = int(my_j / 40)
        for my_k in range(0, 13):
            X_adv2 = my_re_noise(X_adv, X_test, mode=1, pos_num=my_j, stride=0.02 * my_k)
            y2, y22, y222 = predict(sess, my_env_1, X_adv)
            y3, y33, y333 = predict(sess, my_env_1, X_adv2)
            z0 = np.argmax(y_test, axis=1)
            z2 = np.argmax(y2, axis=1)
            z3 = np.argmax(y3, axis=1)
            z22 = np.argmax(y22, axis=1)
            z33 = np.argmax(y33, axis=1)
            z222 = np.argmax(y222, axis=1)
            z333 = np.argmax(y333, axis=1)
            z666 = [z0, z2, z22, z222]
            a1 = my_HD(z2, z3)
            print('rateA = {0}, stride = {1}, distance = {2}, 扰动率 = {3}, 扰动强度= {4}'.format(my_j, my_k, a1, my_j / 784,
                                                                                           my_k / 12))
            # my_diverge[tmp, my_k] = a1
    my_fig = plt.figure()
    ax = my_fig.gca(projection='3d')
    sample_n = np.arange(0, 520, 40)
    soundness = np.arange(0, 13, 1)
    surf_x, surf_y = np.meshgrid(sample_n, soundness)
    surf = ax.plot_surface(surf_x, surf_y, my_diverge)
    plt.show()

    # 第二种方式 stride=eps pos_num=epochs * rateA
    # 但这种方式很可能自己的噪声就相互抵消了，应该和他一样，让总的噪声朝一个方向
    # 那就需要把符号去了

    return 0


def voting_algorithm_use_softmax(z2, z22, z222, count: int):
    """
    投票机制。使用最终 softmax 结果
    如果三个结果都不同的话，标记为-1
    @param z2:
    @param z22:
    @param z222:
    @param count: 数据量
    @return: 投票结果数量，已经进行softmax之后的，最终结果
    """
    # 直接的投票结果矩阵
    voting_result_matrix = np.zeros(count, )
    for i in range(0, count):
        temp1 = z2[i]
        temp2 = z22[i]
        temp3 = z222[i]
        # 统计两个数组相同元素个数(使用集合去重）
        same_num_count = len(set([temp1, temp2, temp3]))
        if same_num_count == 1:
            voting_result_matrix[i] = temp1
            # test_probability_array[i][max_index] = maximum_probability

        elif same_num_count == 2:
            if temp1 == temp2 or temp1 == temp3:
                voting_result_matrix[i] = temp1
                # print("one not same")
                # test_probability_array[i][max_index] = maximum_probability
            else:
                voting_result_matrix[i] = temp2
                # test_probability_array[i][max_index] = maximum_probability
        else:
            # voting_result_matrix[i] = max_index
            # test_probability_array[i][max_index] = maximum_probability
            voting_result_matrix[i] = -1
            # print("three not same")
    return voting_result_matrix


def voting_algorithm_if_three_not_same_use_probability(y2, y22, y222, z2, z22, z222, count):
    """
    投票机制。使用最终 softmax 结果
    如果三个结果都不同的话，使用原始概率计算所得的结果
    @param y2:
    @param y22:
    @param y222:
    @param z2:
    @param z22:
    @param z222:
    @param count: 样本数量
    @return: 投票结果数量，已经进行softmax之后的，最终结果
    """
    # 直接的投票结果矩阵
    voting_result_matrix = np.zeros(count, )
    for i in range(0, count):
        temp1 = z2[i]
        temp2 = z22[i]
        temp3 = z222[i]
        # 统计两个数组相同元素个数(使用集合去重）
        same_num_count = len(set([temp1, temp2, temp3]))
        if same_num_count == 1:
            voting_result_matrix[i] = temp1
            # test_probability_array[i][max_index] = maximum_probability

        elif same_num_count == 2:
            if temp1 == temp2 or temp1 == temp3:
                voting_result_matrix[i] = temp1
                # print("one not same")
                # test_probability_array[i][max_index] = maximum_probability
            else:
                voting_result_matrix[i] = temp2
                # test_probability_array[i][max_index] = maximum_probability
        # 三个都不同
        else:
            # voting_result_matrix[i] = max_index
            # test_probability_array[i][max_index] = maximum_probability
            temp1 = y2[i]
            temp2 = y22[i]
            temp3 = y222[i]

            max1 = np.max(temp1)
            index1 = np.argmax(temp1)

            max2 = np.max(temp2)
            index2 = np.argmax(temp1)

            max3 = np.max(temp3)
            index3 = np.argmax(temp1)

            # 最大概率
            maximum_probability = max([max1, max2, max3])
            # 最大概率的下标
            max_index = [index1, index2, index3][np.argmax([max1, max2, max3])]

            voting_result_matrix[i] = max_index

            # print("three not same")
    return voting_result_matrix


def voting_algorithm_without_softmax(y2, y22, y222, count: int):
    """
    投票算法：直接使用概率进行计算，发现结果100%正确
    自定义 softmax 算法可能存在某些不明原因导致这种不正确的结果
    @param y2:
    @param y22:
    @param y222:
    @return: 直接的最终结果，处理后的概率向量
    """
    test_probability_array = np.zeros((count, 10))
    voting_result_matrix = np.zeros(count, )
    for i in range(0, count):
        temp1 = y2[i]
        temp2 = y22[i]
        temp3 = y222[i]

        # 模型一的输出向量的最大概率与相应的 index
        max1 = np.max(temp1)
        index1 = np.argmax(temp1)

        max2 = np.max(temp2)
        index2 = np.argmax(temp1)

        max3 = np.max(temp3)
        index3 = np.argmax(temp1)

        # 最大概率
        maximum_probability = max([max1, max2, max3])
        # 最大概率的下标
        max_index = [index1, index2, index3][np.argmax([max1, max2, max3])]

        # 统计两个数组相同元素个数(使用集合去重）
        same_num_count = len(set([index1, index2, index3]))
        if same_num_count == 1:
            voting_result_matrix[i] = index1
            test_probability_array[i][max_index] = maximum_probability

        elif same_num_count == 2:
            if index1 == index2 or index1 == index3:
                voting_result_matrix[i] = index1
                test_probability_array[i][max_index] = maximum_probability
            else:
                voting_result_matrix[i] = index2
                test_probability_array[i][max_index] = maximum_probability
        else:
            voting_result_matrix[i] = max_index
            test_probability_array[i][max_index] = maximum_probability

    return voting_result_matrix, test_probability_array


def calculate_run_time(function_name: str, start_time: datetime, end_time: datetime):
    """
    运行时间计算函数
    @param function_name:
    @param start_time:
    @param end_time:
    @return:
    """
    print("\nstart time of:", function_name, start_time)
    print("\nend time of:", function_name, end_time)
    print("\nTotal time:", (end_time - start_time).seconds, " s")


if __name__ == "__main__":
    program_start_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    # entropy = pow(2, 6272)

    # 数据读取部分

    fake1 = np.random.random_integers(0, 255, size=(10000, 28, 28, 1))
    fake2 = fake1.astype(np.float32) / 255
    # plt.figure()
    # plt.imshow(fake2[2], cmap='gray')
    # plt.show()
    print('\nLoading MNIST')
    mnist = tf.keras.datasets.mnist

    # 数据处理部分
    img_size = 28
    img_chan = 1
    n_classes = 10

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = np.reshape(X_train, [-1, img_size, img_size, img_chan])
    X_train = X_train.astype(np.float32) / 255

    X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
    X_test = X_test.astype(np.float32) / 255

    to_categorical = tf.keras.utils.to_categorical

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print('\nSpliting data')
    ind = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[ind], y_train[ind]

    VALIDATION_SPLIT = 0.1
    n = int(X_train.shape[0] * (1 - VALIDATION_SPLIT))

    X_valid = X_train[n:]
    X_train = X_train[:n]
    y_valid = y_train[n:]
    y_train = y_train[:n]

    # 训练逻辑部分
    my_env_1 = Dummy()

    setting_up_env(my_env_1)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # 训练部分

    # #正常训练部分
    print('\nTraining')

    # todo 训练与对抗训练计时操作
    train_start_time = datetime.datetime.now()

    train(sess, my_env_1, X_train, y_train, X_valid, y_valid, load=False, epochs=5,
          name='mnist')

    train_end_time = datetime.datetime.now()
    calculate_run_time("train", train_start_time, train_end_time)

    print('\n Evaluating on clean data(The final loss)')

    evaluate_start_time = datetime.datetime.now()

    evaluate(sess, my_env_1, X_test, y_test)

    evaluate_end_time = datetime.datetime.now()
    calculate_run_time("Evaluating on clean data(The final loss)", evaluate_start_time, evaluate_end_time)


    print('\n Generating adversarial data')
    print("epochs = ", 11)
    print("eps = ", 0.02)
    print("sum of eps = ", 0.02 * (11))

    # #对抗训练

    x_adv_start_time = datetime.datetime.now()
    X_adv = make_fgmt(sess, my_env_1, X_test, eps=0.02, epochs=11)
    x_adv_end_time = datetime.datetime.now()
    calculate_run_time("x_adv", x_adv_start_time, x_adv_end_time)

    print('\n Evaluating on adversarial data(The final loss)')

    evaluate_start_time = datetime.datetime.now()
    evaluate(sess, my_env_1, X_adv, y_test)
    evaluate_end_time = datetime.datetime.now()
    calculate_run_time("Evaluating on adversarial data(The final loss)", evaluate_start_time, evaluate_end_time)

    print('\nRandomly sample adversarial data from each category')

    # 重扰动阶段
    # X_adv1 = my_re_noise(X_adv, X_test, mode=1, pos_num=1, stride=0.18, dire=0)

    # my_single_pixel(X_adv, X_test, my_batch_size=100000)
    # my_noise_shuffle(X_adv, X_test)
    # my_make_re_noise(X_adv, X_test)

    # todo 序列化保存数据，以供后续分析

    try:
        folder = os.path.exists(program_start_time)
        if not folder:
            os.makedirs(program_start_time, exist_ok=True)
    except Exception as e:
        print(e.__str__())

    y1, y11, y111 = predict(sess, my_env_1, X_test)
    y2, y22, y222 = predict(sess, my_env_1, X_adv)

    # y3, y33, y333 = predict(sess, my_env_1, X_adv1)
    # 原始标签
    z0 = np.argmax(y_test, axis=1)
    # 第一个模型的预测标签（在干净的数据集上得到的结果）
    z1 = np.argmax(y1, axis=1)
    # 第一个模型的结果（攻击之后的结果）
    z2 = np.argmax(y2, axis=1)
    # z3 = np.argmax(y3, axis=1)
    # 第二个模型的结果
    z11 = np.argmax(y11, axis=1)
    z22 = np.argmax(y22, axis=1)
    # z33 = np.argmax(y33, axis=1)
    # 第三个模型的结果
    z111 = np.argmax(y111, axis=1)
    z222 = np.argmax(y222, axis=1)
    # z333 = np.argmax(y333, axis=1)
    z666 = [z0, z2, z22, z222]
    # a1 = my_HD(z2, z3)
    a2 = my_HD(z2, z22)
    a3 = my_HD(z2, z222)

    # todo 投票机制：（应该需要写多个函数）

    # 普通的多数投票机制
    # 1. 3个投票，如果有多数，则取多数值，
    # 2. 如果没有多数，则取各个模型中概率值最大的结果
    # 线性组合？那么应该是需要添加一个缓存数组，根据线性组合处理三个模型的输出，然后得出输出结果矩阵
    # 根据得出的结果重组矩阵，选出最大值
    # 百分比
    # #剔除-1再算另一个百分比
    # #

    z_voting_use_softmax = voting_algorithm_use_softmax(z2, z22, z222, 10000)

    z_voting_use_softmax_if_three_not_same_use_probability = voting_algorithm_if_three_not_same_use_probability(y2,
                                                                                                                y22,
                                                                                                                y222,
                                                                                                                z2,
                                                                                                                z22,
                                                                                                                z222,
                                                                                                                10000)

    z_voting_without_softmax, y_voting_without_softmax = voting_algorithm_without_softmax(y2, y22, y222, 10000)

    z_voting_use_softmax_if_three_not_same_use_probability_count = 0
    z_voting_use_softmax_count = 0
    z_voting_use_softmax_count_remove_1 = 0
    useless_data_count = 0
    z_voting_without_softmax_count = 0

    z2_count = 0
    z22_count = 0
    z222_count = 0

    z1_count = 0
    z11_count = 0
    z111_count = 0

    for i in range(0, 10000):
        if z0[i] == z1[i]:
            z1_count += 1
        if z0[i] == z11[i]:
            z11_count += 1
        if z0[i] == z111[i]:
            z111_count += 1

    print("z1 的准确率为：", float(z1_count) / 10000.0 * 100.0, "%")
    print("z11 的准确率为：", float(z11_count) / 10000.0 * 100.0, "%")
    print("z111 的准确率为：", float(z111_count) / 10000.0 * 100.0, "%")

    for i in range(0, 10000):
        if z0[i] == z2[i]:
            z2_count += 1
        if z0[i] == z22[i]:
            z22_count += 1
        if z0[i] == z222[i]:
            z222_count += 1

    print("z2 的准确率为：", float(z2_count) / 10000.0 * 100.0, "%")
    print("z22 的准确率为：", float(z22_count) / 10000.0 * 100.0, "%")
    print("z222 的准确率为：", float(z222_count) / 10000.0 * 100.0, "%")

    # 累加求和，准备计算概率
    for i in range(0, 10000):
        # 计算拥有-1的准确率
        if (z0[i] == z_voting_use_softmax[i]):
            z_voting_use_softmax_count += 1
        # 计算去除-1之后的准确率
        if z_voting_use_softmax[i] == -1:
            useless_data_count += 1
        elif z0[i] == z_voting_use_softmax[i]:
            z_voting_use_softmax_count_remove_1 += 1
        # 计算在3个结果都不同时，使用概率最大值填充的准确率
        if z0[i] == z_voting_use_softmax_if_three_not_same_use_probability[i]:
            z_voting_use_softmax_if_three_not_same_use_probability_count += 1
        # 计算只使用概率的情况下的准确率
        if z0[i] == z_voting_without_softmax[i]:
            z_voting_without_softmax_count += 1

    print("计算使用 softmax ，拥有-1的准确率:\r\n", float(z_voting_use_softmax_count) / 10000.0 * 100.0, "%")
    print("计算使用 softmax ，去除-1之后的准确率:\r\n",
          float(z_voting_use_softmax_count_remove_1) / (10000.0 - useless_data_count) * 100.0, "%")
    print("计算在3个结果都不同时，使用概率最大值填充的准确率:\r\n",
          float(z_voting_use_softmax_if_three_not_same_use_probability_count) / 10000.0 * 100.0, "%")
    print("计算只使用概率的情况下的准确率:\r\n", float(z_voting_without_softmax_count) / 10000.0 * 100.0, "%")

    """
    z1 的准确率为： 99.11999999999999 %
    z11 的准确率为： 98.77 %
    z111 的准确率为： 98.83 %
    z2 的准确率为： 11.559999999999999 %
    z22 的准确率为： 65.68 %
    z222 的准确率为： 64.35 %
    计算使用 softmax ，拥有-1的准确率:
     55.97 %
    计算使用 softmax ，去除-1之后的准确率:
     64.22260470453242 %
    计算在3个结果都不同时，使用概率最大值填充的准确率:
     56.00000000000001 %
    计算只使用概率的情况下的准确率:
     11.559999999999999 %
    """
