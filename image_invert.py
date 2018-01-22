import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import random
import scipy
np.set_printoptions(threshold=np.nan)
import vgg



#################################
# parameters
max_tries = 10000    
pooling = 'avg'


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("image", "./monkey.jpg", """Image file path""")
tf.flags.DEFINE_string("vgg19", "./imagenet-vgg-verydeep-19.mat", """Pre-trained VGG19 file path""")
tf.flags.DEFINE_string("invert_layer", "conv5_1", """Layer to invert from.""")
sample_dir = './samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

from shutil import copyfile
copyfile(__file__, sample_dir+'/'+__file__)



def read_image(file, scale_w=0, scale_h=0):
    img = scipy.misc.imread(file, mode='RGB').astype(np.float32)
    if (scale_w*scale_h):
        img = scipy.misc.imresize(img, [scale_w, scale_h])
    return img


def main(_):


    global_step = tf.Variable(0, trainable=False, name='global_step')
    invert_layer = FLAGS.invert_layer

    ### Load pre-trained VGG wieghts
    vgg_mat_file = FLAGS.vgg19
    print ("pretrained-VGG : {}".format(FLAGS.vgg19))
    vgg_weights, vgg_mean_pixel = vgg.load_net(vgg_mat_file)
    print ("vgg_mean_pixel : ", vgg_mean_pixel)
    
    ### Read input image 
    image = FLAGS.image
    print ("input image : {}".format(FLAGS.image))
    img = read_image(image, 224, 224)
    scipy.misc.imsave(sample_dir+'/input_image.png', img)
            
    img = img - vgg_mean_pixel
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)   # extend shape for VGG input
    img_shape=np.shape(img)
    print ("Image shape : ", np.shape(img))
    
    
    gpu_options = tf.GPUOptions(allow_growth=True)  
    ### Comput content feature of 'invert_layer'
    X_content_feature={}
    content_graph = tf.Graph()
    with content_graph.as_default():
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        X_content = tf.placeholder('float32', shape=img_shape)
        network = vgg.net_preloaded(vgg_weights, img, pooling)
        X_content_feature = sess.run(network[invert_layer], feed_dict={X_content:img})


    ### Define network to learn 'X'
    # X_sigma = tf.norm(vgg_mean_pixel)*img_shape[1]   # roughly...
    # X_sigma = tf.cast(X_sigma, tf.float32)
    # X = tf.Variable(tf.random_normal(img_shape))*X_sigma
    X = tf.Variable(tf.random_normal(img_shape))
    invert_net = vgg.net_preloaded(vgg_weights, X, pooling)
    X_invert_feature = invert_net[invert_layer]
    
    l2_loss = tf.norm(X_content_feature-X_invert_feature, 'euclidean')/tf.norm(X_content_feature, 'euclidean')
    #total_variation_loss = tf.image.total_variation(img+X)[0]
    total_variation_loss = tf.reduce_sum(tf.image.total_variation(tf.convert_to_tensor(img+X)))
    sigma_tv = 5e-7
    loss = l2_loss + sigma_tv*total_variation_loss

    train_step = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.5).minimize(loss, global_step = global_step)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    
    for step in range(max_tries):  
              
        _ , _loss = sess.run([train_step, loss])      
        print("step: %06d"%step, "loss: {:.04}".format(_loss) )
        #_tv = sess.run(total_variation_loss)
        #print("total_variation_loss : ", sigma_tv*_tv)
                   
        # testing
        if not (step+1)%100: 
            this_X = sess.run(X)
            this_X = this_X + vgg_mean_pixel
            scipy.misc.imsave(sample_dir+'/invert_{}'.format(str(step+1).zfill(6)) + '.png', this_X[0])
            '''
            this_X = this_X/255
            fig, ax = plt.subplots(1, 1, figsize=(1,1), dpi=400)
            ax.set_axis_off()
            ax.imshow(this_X[0])
            plt.savefig(sample_dir+'/invert_{}'.format(str(step).zfill(6)) + '.png', bbox_inches='tight')
            plt.close(fig)
           '''
    

if __name__ == "__main__":
    tf.app.run()



