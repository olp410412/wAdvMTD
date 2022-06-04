import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import layers
from sklearn.metrics import classification_report
import warnings
import time


#模型建立
def conv2d_bn(inpt, filters=64, kernel_size=(3,3), strides=1, padding='same'):
    '''卷积、归一化和relu三合一'''
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(inpt)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def basic_bottle(inpt, filters=64, kernel_size=(3,3), strides=1, padding='same', if_baisc=False):
    '''18中的4个basic_bottle'''
    x = conv2d_bn(inpt, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
    x = conv2d_bn(x, filters=filters)
    if if_baisc==True:
        temp = conv2d_bn(inpt, filters=filters, kernel_size=(1,1), strides=2, padding='same')
        outt = layers.add([x, temp])
    else:
        outt = layers.add([x, inpt])
    return outt

def resnet18(class_nums):
    '''主模型'''
    inpt = layers.Input(shape=(28,28,1))
    #layer 1
    x = conv2d_bn(inpt, filters=64, kernel_size=(7,7), strides=2, padding='valid')
    x = layers.MaxPool2D(pool_size=(3,3), strides=2)(x)
    #layer 2
    x = basic_bottle(x, filters=64, kernel_size=(3,3), strides=1, padding='same', if_baisc=False)
    x = basic_bottle(x, filters=64, kernel_size=(3,3), strides=1, padding='same', if_baisc=False)
    #layer 3
    x = basic_bottle(x, filters=128, kernel_size=(3, 3), strides=2, padding='same', if_baisc=True)
    x = basic_bottle(x, filters=128, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    # layer 4
    x = basic_bottle(x, filters=256, kernel_size=(3, 3), strides=2, padding='same', if_baisc=True)
    x = basic_bottle(x, filters=256, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    # layer 5
    x = basic_bottle(x, filters=512, kernel_size=(3, 3), strides=2, padding='same', if_baisc=True)
    x = basic_bottle(x, filters=512, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    #GlobalAveragePool
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(class_nums, activation='softmax')(x)
    model = tf.keras.Model(inputs=inpt, outputs=x)
    return model


def resnet18_alter_1(class_nums):
    '''主模型'''
    inpt = layers.Input(shape=(28, 28, 1))
    # my layer 0
    x = conv2d_bn(inpt, filters=64, kernel_size=(7, 7), strides=2, padding='valid')
    x = layers.MaxPool2D(pool_size=(3, 3), strides=2)(x)
    # layer 1
    x = basic_bottle(x, filters=64, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    x = basic_bottle(x, filters=64, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    # layer 2
    x = basic_bottle(x, filters=128, kernel_size=(3, 3), strides=2, padding='same', if_baisc=True)
    x = basic_bottle(x, filters=128, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    # layer 3
    x = basic_bottle(x, filters=256, kernel_size=(3, 3), strides=2, padding='same', if_baisc=True)
    x = basic_bottle(x, filters=256, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    # layer 4
   # x = basic_bottle(x, filters=256, kernel_size=(3, 3), strides=2, padding='same', if_baisc=True)
   # x = basic_bottle(x, filters=256, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    # layer 5
   # x = basic_bottle(x, filters=512, kernel_size=(3, 3), strides=2, padding='same', if_baisc=True)
   # x = basic_bottle(x, filters=512, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    # GlobalAveragePool
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(class_nums, activation='softmax')(x)
    model1 = tf.keras.Model(inputs=inpt, outputs=x)
    return model1


def make_fgmt(sess, env, X_data, epochs=1, eps=0.01, batch_size=64):
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
        adv = sess.run(env.x_fgmt, feed_dict={
            env.x: X_data[start:end],
            env.fgsm_eps: eps,
            env.fgsm_epochs: epochs})
        X_adv[start:end] = adv
    print()

    return X_adv


def fgmt(model, x, y=None, eps=0.01, epochs=1, sign=True, clip_min=0.,
         clip_max=1.):
    """
    Fast gradient method with target

    See https://arxiv.org/pdf/1607.02533.pdf.  This method is different from
    FGM that instead of decreasing the probability for the correct label, it
    increases the probability for the desired label.

    :param model: A model that returns the output as well as logits.
    :param x: The input placeholder.
    :param y: The desired target label, set to the least-likely class if None.
    :param eps: The noise scale factor.
    :param epochs: Maximum epoch to run.
    :param sign: Use gradient sign if True, otherwise gradient values.
    :param clip_min: Minimum value in output.
    :param clip_max: Maximum value in output.
    """
    xadv = tf.identity(x)

    ybar0 = model(xadv)
    ybar = tf.nn.softmax(ybar0)
    # ybar = np.argmax(ybar0)
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

    def _cond(xadv, i):
        return tf.less(i, epochs)

    def _body(xadv, i):
        logits = model(xadv)
        ybar = np.argmax(logits)
        loss = loss_fn(labels=target, logits=logits)
        dy_dx, = tf.gradients(loss, xadv)
        xadv = tf.stop_gradient(xadv + eps*noise_fn(dy_dx))
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)
        return xadv, i+1

    xadv, _ = tf.while_loop(_cond, _body, (xadv, 0), back_prop=False,
                            name='fast_gradient_target')
    return xadv


class Dummy:
    pass


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    img_size = 28
    img_chan = 1
    n_classes = 10

    print('\nLoading MNIST')

    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.reshape(X_train, [-1, img_size, img_size, img_chan])
    X_train = X_train.astype(np.float32) / 255
    X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
    X_test = X_test.astype(np.float32) / 255
    # 选择keras时，有时候不用one-hot
    # to_categorical = tf.keras.utils.to_categorical
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)

    print('\nSpliting data')

    ind = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[ind], y_train[ind]

    VALIDATION_SPLIT = 0.1
    n = int(X_train.shape[0] * (1 - VALIDATION_SPLIT))
    X_valid = X_train[n:]
    X_train = X_train[:n]
    y_valid = y_train[n:]
    y_train = y_train[:n]

    model = resnet18_alter_1(10)
    env = Dummy()
    with tf.variable_scope('model233'):
        env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                               name='x')
        env.y = tf.placeholder(tf.float32, (None, n_classes), name='y233')
        env.training = tf.placeholder_with_default(False, (), name='mode233')
        # env.ybar, logits = model(env.x, logits=True, training=env.training)

        # with tf.variable_scope('acc'):
        #    count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
        #    env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

        # with tf.variable_scope('loss'):
        #     xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
        #                                                   logits=logits)
        #     env.loss = tf.reduce_mean(xent, name='loss')

        # with tf.variable_scope('train_op'):
        #    optimizer = tf.train.AdamOptimizer()
        #    env.train_op = optimizer.minimize(env.loss)

        # env.saver = tf.train.Saver()
    with tf.variable_scope('model233', reuse=True):
        env.fgsm_eps = tf.placeholder(tf.float32, (), name='fgsm_eps233')
        env.fgsm_epochs = tf.placeholder(tf.int32, (), name='fgsm_epochs233')
        env.x_fgmt = fgmt(model, env.x, epochs=env.fgsm_epochs, eps=env.fgsm_eps)
    print('\nInitializing graph')
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # 开始训练


    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    t0 = time.time()
    model.fit(X_train, y_train, batch_size=64, epochs=3, validation_split=0.2)  # epochs可以调的大一点，如果你机器好的话
    t1 = time.time()

    # -----
    model1 = resnet18_alter_1(10)
    # model = resnet18(10)
    model1.summary()
    model1.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    t01 = time.time()
    model1.fit(X_train, y_train, batch_size=64, epochs=3, validation_split=0.2)  # epochs可以调的大一点，如果你机器好的话
    t11 = time.time()

    # ------



    # 验证结果
    pred_y00 = model.predict(X_test)
    pred_y01 = model1.predict(X_test)
    # pred_y 是logits形式，因为后面有新的argmax
    last_pred00 = []
    last_pred01 = []
    if pred_y00.shape[0] != pred_y00.shape[0]:
        print("shape error on pred_y00")
    for i in range(pred_y00.shape[0]):
        last_pred00.append(pred_y00[i].argmax())
        last_pred01.append(pred_y01[i].argmax())
    last_pred00 = np.array(last_pred00)
    last_pred01 = np.array(last_pred01)
    print(classification_report(y_test, last_pred00))
    print(classification_report(y_test, last_pred01))

    print("-----------------------------------------------")


    t2 = time.time()

    X_adv = make_fgmt(sess, env, X_test, eps=0.16, epochs=1)

    t3 = time.time()

    print("-----------------------------------------------")

    pred_on_adv = model.predict(X_adv)
    last_pred1 = []
    for i in range(pred_on_adv.shape[0]):
        last_pred1.append(pred_on_adv[i].argmax())
    last_pred1 = np.array(last_pred1)
    print(classification_report(y_test, last_pred1))

    pred_on_model1 = model1.predict(X_adv)
    last_pred2 = []
    for i in range(pred_on_model1.shape[0]):
        last_pred2.append(pred_on_model1[i].argmax())
    last_pred2 = np.array(last_pred2)
    print(classification_report(y_test, last_pred2))

    #
    print("T" + str(t1 - t0) + "    " + str(t3 - t2))
    print("233")


