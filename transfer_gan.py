import os, sys
sys.path.append(os.getcwd())

import time
import functools

import numpy as np
import tensorflow as tf
#import sklearn.datasets

import tflib as lib
import cPickle as pickle
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.lsun
import tflib.ops.layernorm
import tflib.plot
import pdb

# fill in the path to the extracted files here!
DATA_DIR = './data/bedroom'
# Path of pretrained model, please select the pretrained models
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_64x64.py!')
MODE = 'wgan-gp' # Name of mode
TARGET_DOMIAN = 'lsun'# Name of target domain 
SOURCE_DOMAIN = 'bedroom'# imagenet, places, celebA, bedroom,

PRETRAINED_MODEL = './transfer_model/%s/wgan-gp.model'%SOURCE_DOMAIN 

DIM = 64 # Model dimensionality
LR = 1e-4# Learning rate, normally reduce it when pretrained model is used for limited data
N_GPUS = 1 # Number of GPUs
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many iterations to train for
BATCH_SIZE = 64 # Batch size. Must be a multiple of N_GPUS
CRITIC_ITERS = 5 # How many iterations to train the critic for
OUTPUT_DIM = 64*64*3 # Number of pixels in each iamge

# Define correspond folders to save result and model
RESULTS = './result/source_%s_target_%s_batch_%d_dim_%d_lr_%s'%(SOURCE_DOMAIN, TARGET_DOMIAN, BATCH_SIZE, DIM, LR)
MODEL_PATH = '%s/model'%RESULTS
SAMPLES_DIR = '%s/samples'%RESULTS
CHECKPOINT = '%s/checkpoint'%RESULTS

# Create directories if necessary
if not os.path.exists(MODEL_PATH):
  print("*** create tboard dir %s" % MODEL_PATH)
  os.makedirs(MODEL_PATH)

if not os.path.exists(SAMPLES_DIR):
  print("*** create sample dir %s" % SAMPLES_DIR)
  os.makedirs(SAMPLES_DIR)

if not os.path.exists(CHECKPOINT):
  print("*** create checkpoint dir %s" % CHECKPOINT )
  os.makedirs(CHECKPOINT)

# Print current setting
lib.print_model_settings(locals().copy())

# Import generator and discriminator 
def GeneratorAndDiscriminator():
    return Generator, Discriminator

DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)

def Normalize(name, axes, inputs):
    if ('Discriminator' in name) and (MODE == 'wgan-gp'):
        if axes != [0,2,3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    else:
        return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)

def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)

def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4*kwargs['output_dim']
    output = lib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    return output

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = MeanPoolConv
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = UpsampleConv
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name+'.BN1', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)
    output = Normalize(name+'.BN2', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)

    return shortcut + output


def save_model(sess, saver, checkpoint_dir, step):
    model_name = "%s.model" % TARGET_DOMIAN
    saver.save(sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)


# ! Generators

def Generator(n_samples, noise=None, dim=DIM, nonlinearity=tf.nn.relu):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*8*dim, noise)
    output = tf.reshape(output, [-1, 8*dim, 4, 4])

    output = ResidualBlock('Generator.Res1', 8*dim, 8*dim, 3, output, resample='up')
    output = ResidualBlock('Generator.Res2', 8*dim, 4*dim, 3, output, resample='up')
    output = ResidualBlock('Generator.Res3', 4*dim, 2*dim, 3, output, resample='up')
    output = ResidualBlock('Generator.Res4', 2*dim, 1*dim, 3, output, resample='up')

    output = Normalize('Generator.OutputN', [0,2,3], output)
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', 1*dim, 3, 3, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

# ! Discriminators

def Discriminator(inputs, dim=DIM):
    output = tf.reshape(inputs, [-1, 3, 64, 64])
    output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3, dim, 3, output, he_init=False)

    output = ResidualBlock('Discriminator.Res1', dim, 2*dim, 3, output, resample='down')
    output = ResidualBlock('Discriminator.Res2', 2*dim, 4*dim, 3, output, resample='down')
    output = ResidualBlock('Discriminator.Res3', 4*dim, 8*dim, 3, output, resample='down')
    output = ResidualBlock('Discriminator.Res4', 8*dim, 8*dim, 3, output, resample='down')

    output = tf.reshape(output, [-1, 4*4*8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    return tf.reshape(output, [-1])

Generator, Discriminator = GeneratorAndDiscriminator()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

    all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, 64, 64])
    if tf.__version__.startswith('1.'):
        split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
    else:
        split_real_data_conv = tf.split(0, len(DEVICES), all_real_data_conv)
    gen_costs, disc_costs = [],[]

    for device_index, (device, real_data_conv) in enumerate(zip(DEVICES, split_real_data_conv)):
        with tf.device(device):
            real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [BATCH_SIZE/len(DEVICES), OUTPUT_DIM])
            fake_data = Generator(BATCH_SIZE/len(DEVICES))

            disc_real = Discriminator(real_data)
            disc_fake = Discriminator(fake_data)

            gen_cost = -tf.reduce_mean(disc_fake)
            disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

            alpha = tf.random_uniform(
                shape=[BATCH_SIZE/len(DEVICES),1], 
                minval=0.,
                maxval=1.
            )
            differences = fake_data - real_data
            interpolates = real_data + (alpha*differences)
            gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            disc_cost += LAMBDA*gradient_penalty
            gen_costs.append(gen_cost)
            disc_costs.append(disc_cost)

    gen_cost = tf.add_n(gen_costs) / len(DEVICES)
    disc_cost = tf.add_n(disc_costs) / len(DEVICES)


    # Generator and Discriminator optimization
    gen_train_op = tf.train.AdamOptimizer(learning_rate=LR, beta1=0., beta2=0.9).minimize(gen_cost,
                                      var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=LR, beta1=0., beta2=0.9).minimize(disc_cost,
                                       var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    # For generating samples
    fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
    all_fixed_noise_samples = []
    for device_index, device in enumerate(DEVICES):
        n_samples = BATCH_SIZE / len(DEVICES)
        all_fixed_noise_samples.append(Generator(n_samples, noise=fixed_noise[device_index*n_samples:(device_index+1)*n_samples]))
    if tf.__version__.startswith('1.'):
        all_fixed_noise_samples = tf.concat(all_fixed_noise_samples, axis=0)
    else:
        all_fixed_noise_samples = tf.concat(0, all_fixed_noise_samples)
    def generate_image(iteration):
        samples = session.run(all_fixed_noise_samples)
        samples = ((samples+1.)*(255.99/2)).astype('int32')
        lib.save_images.save_images(samples.reshape((BATCH_SIZE, 3, 64, 64)), '%s/samples_%d.png'%(SAMPLES_DIR, iteration))

    # Dataset iterator
    train_gen, dev_gen = tflib.lsun.load(BATCH_SIZE, data_dir=DATA_DIR)

    def inf_train_gen():
        while True:
            for (images,) in train_gen():
                yield images

    # Save a batch of ground-truth samples
    _x = inf_train_gen().next()
    _x_r = session.run(real_data, feed_dict={real_data_conv: _x[:BATCH_SIZE/N_GPUS]})
    _x_r = ((_x_r+1.)*(255.99/2)).astype('int32')
    lib.save_images.save_images(_x_r.reshape((BATCH_SIZE/N_GPUS, 3, 64, 64)), '%s/samples_groundtruth.png'%SAMPLES_DIR)


    session.run(tf.initialize_all_variables())
    if False:
        var_list = pickle.load(open('./all_var'))

        for name,value in var_list.iteritems():
            #Bool = np.asarray([i for i in lib._params[name[7:]].get_shape().as_list()]) 
            #if Bool == value[0].shape:
            session.run(lib._params[name[7:]].assign(value[0]))
            print name[7:]
            #else:
            #    pdb.set_trace()
            #    session.run(lib._params[name].assign(np.tile(value[0],[10,1])))

    # Restore pretrained model
    saver = tf.train.Saver(lib.params_with_name('Generator') + lib.params_with_name('Discriminator.'))
    saver.restore(session, PRETRAINED_MODEL)
    # How many  model to save
    saver = tf.train.Saver(max_to_keep = 1000)
    gen = inf_train_gen()

    generate_image(0)
    for iteration in xrange(ITERS):

        start_time = time.time()

        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op)

        # Train critic
        disc_iters = CRITIC_ITERS
        for i in xrange(disc_iters):
            _data = gen.next()
            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={all_real_data_conv: _data})

        lib.plot.plot('%s/train disc cost'%SAMPLES_DIR, _disc_cost)
        lib.plot.plot('%s/time'%SAMPLES_DIR, time.time() - start_time)

        if iteration % 200 == 199:
            t = time.time()
            dev_disc_costs = []
            for (images,) in dev_gen():
                _dev_disc_cost = session.run(disc_cost, feed_dict={all_real_data_conv: images}) 
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('%s/dev disc cost'%SAMPLES_DIR, np.mean(dev_disc_costs))

        if iteration % 200 == 0:
            generate_image(iteration)

        if (iteration < 5) or (iteration % 200 == 199):
            lib.plot.flush(SAMPLES_DIR)

        lib.plot.tick()
        if iteration % 200 == 0:
            save_model(session, saver, MODEL_PATH, iteration)
