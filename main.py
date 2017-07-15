import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables, generate_image_from_z

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")

# Custom flags
flags.DEFINE_string("folder_suffix", "", "Suffix of folder names (logs, samples, checkpoints), []")
flags.DEFINE_string("inference_z_src", "_", "Name of checkpoint file suffix to generate image from []")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_integer("GpD_ratio", -1, "Generator per discriminator training iteration steps. -1 uses the controller [-1]")
flags.DEFINE_string("blur_strategy", "reg_hyp", "Blur strategy to use for training. [None, 3x3, reg_lin, reg_hyp]")
flags.DEFINE_boolean("with_overlay", False, "If True, an overlay indicating the realism from the discriminators point of view will be added to all images in a batch. Additionally each batch will be sorted using this realism. Most realistic image on bottom right. [False]")
flags.DEFINE_integer("sample_every", 1000, "Sample a new image all x iterations during training. [1000]")
flags.DEFINE_boolean("with_loose_encoder", False, "If True, the experimental loose encoder network will be added to the model. This can improve image quality significantly. [False]")

# For the Controller
flags.DEFINE_float("target_starting_G_quality", 0.25, "Target quality of generator in the beginning of the training for the controller [0.25]")
flags.DEFINE_float("target_ending_G_quality", 0.25, "Target quality of generator in the end of the training for the controller [0.25]")
flags.DEFINE_float("control_gain", 0.001, "Gain of the controller [0.001]")

FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height
  
  folder_suffix = FLAGS.folder_suffix
  checkpoint_dir = FLAGS.checkpoint_dir + folder_suffix
  sample_dir = FLAGS.sample_dir + folder_suffix
  log_dir = './logs' + folder_suffix

  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True
  run_config.allow_soft_placement=True

  with tf.Session(config=run_config) as sess:
    if FLAGS.dataset == 'mnist':
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          y_dim=10,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=checkpoint_dir,
          sample_dir=sample_dir,
					log_dir=log_dir,
          blur_strategy=FLAGS.blur_strategy,
          loose_encoder=FLAGS.with_loose_encoder)
    else:
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=checkpoint_dir,
          sample_dir=sample_dir,
					log_dir=log_dir,
          blur_strategy=FLAGS.blur_strategy,
          loose_encoder=FLAGS.with_loose_encoder)

    show_all_variables()

    if FLAGS.train:
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")
      

    # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
    #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
    #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
    #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
    #                 [dcgan.h4_w, dcgan.h4_b, None])
  
      generate_image_from_z(sess, dcgan, FLAGS)

    # Below is codes for visualization
    #OPTION = 1
    #visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
  tf.app.run()
