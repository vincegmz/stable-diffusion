import gc
import io
import os
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions
import evaluation
from torch.utils import tensorboard
from absl import app
from absl import flags
import glob
from omegaconf import OmegaConf
import argparse
from PIL import Image


def evaluate(config,
             workdir,
             eval_folder="eval"):
  """Evaluate trained models.
  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)
  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)
  begin_ckpt = config.eval.begin_ckpt
  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    # Generate samples and compute IS/FID/KID when enabled
    logging.info(eval_dir)
    sample_dir = eval_dir
    dirs = glob.glob(sample_dir)
    if len(dirs) == 0:
      break
    # Directory to save samples. Different for each host to avoid writing conflicts
    for this_sample_dir in dirs:
      sample_paths = glob.glob(os.path.join(this_sample_dir, f"samples_*.npz"))
      sample_paths.sort()
      for sample_path in sample_paths:
        r = os.path.basename(sample_path).split("samples_")[1].split(".npz")[0]
        logging.info(f"evaluation -- ckpt: {ckpt}, round: {r}")
        # Read samples to disk or Google Cloud Storage
        samples = np.load(sample_path, "wb")['samples']
        # resize size of synthetic images to that of dataset
        resized_samples = np.empty((0,config.data.image_size,config.data.image_size,3))
        if config.data.output_size != config.data.image_size:
          for i in range(samples.shape[0]):
            img = Image.fromarray(samples[i])
            resized_samples = np.concatenate([resized_samples,
                                              np.expand_dims(np.array(img.resize([config.data.image_size]*2)),axis = 0)],axis = 0)
        samples = resized_samples
        # Force garbage collection before calling TensorFlow code for Inception network
        gc.collect()
        latents = evaluation.run_inception_distributed(samples, inception_model,
                                                        inceptionv3=inceptionv3)
        # Force garbage collection again before returning to JAX code
        gc.collect()
        # Save latent represents of the Inception network to disk or Google Cloud Storage
        with tf.io.gfile.GFile(os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(
            io_buffer, pool_3=latents["pool_3"])
          fout.write(io_buffer.getvalue())
    # Compute inception scores, FIDs and KIDs.
    # Load all statistics that have been previously computed and saved for each host
    all_pools = []
    for this_sample_dir in dirs:
      stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
      for stat_file in stats:
          with tf.io.gfile.GFile(stat_file, "rb") as fin:
              stat = np.load(fin)
              all_pools.append(stat["pool_3"])
    all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]
    # Load pre-computed dataset statistics.
    data_stats = evaluation.load_dataset_stats(config)
    data_pools = data_stats["pool_3"]
    fid = tfgan.eval.frechet_classifier_distance_from_activations(
      data_pools, all_pools)
    logging.info(
      "ckpt-%d --- FID: %.6e" % (
        ckpt, fid))
    with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}.npz"),
                            "wb") as f:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, fid=fid)
      f.write(io_buffer.getvalue())
def compute_dataset_stats(config):
    '''
    run inception model on the dataset, store the latents[pool_3]
    Input: dataset_path
    output None
    '''
    dataset_path = os.path.join(config.data.data_root,'imgs')
    if not os.path.isdir(dataset_path):
      raise FileNotFoundError('Incorrect data path')
    dataset = np.empty((0,config.data.image_size,config.data.image_size,3))
    count = 0
    for file in os.listdir(dataset_path):
      if count == config.eval.num_samples:
         break
      img = Image.open(os.path.join(dataset_path,file))
      dataset = np.concatenate([dataset,np.expand_dims(np.array(img),0)],axis = 0)
      count+=1

    dataset = np.array(dataset)
    gc.collect()
    inceptionv3 = config.data.image_size >= 256
    latents = evaluation.run_inception_distributed(dataset, evaluation.get_inception_model(inceptionv3=inceptionv3),
                                                        inceptionv3=inceptionv3)
    with tf.io.gfile.GFile(os.path.join(config.data.data_root, f"statistics_cifar.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(
            io_buffer, pool_3=latents["pool_3"])
          fout.write(io_buffer.getvalue())
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',type = str,default ='configs/eval/eval.yaml')
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    compute_dataset_stats(config)
    evaluate(config, config.work_dir,config.eval.eval_folder)

if __name__ == "__main__":
   main()

