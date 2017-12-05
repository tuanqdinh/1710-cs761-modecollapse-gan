import tensorflow as tf
import numpy as np
import os, sys
from scipy.stats import entropy

from vgan import VGAN
from helpers import inf_train_gen, get_dist, classify_dist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# All params
trainning = True
testing = True
avai = False
# constant
DATASET = '1200D'
ds_folder = '../dataset/1200D/'

model_folder = '../out/models/'
inp_path = os.path.abspath('../out/input/')
out_path = os.path.abspath('../out/train/')
sample_file = '../out/synthetic_samples.npy'
# Training
batch_size = 64
n_iters = 20000
print_counter = 500
#Testing
n_samples = 2500


def create_folder(folder_name):
    if not os.path.exists(folder_name):
      os.makedirs(folder_name)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        trainning = eval(sys.argv[1])
        testing = eval(sys.argv[2])
        avai = eval(sys.argv[3])

    gg = VGAN(model_folder)

    if trainning == True:
        print("Training...")
        create_folder(model_folder)
        create_folder(out_path)
        create_folder(inp_path)
        #Dataset
        gen = inf_train_gen(DATASET, batch_size)
        gg.train(gen, batch_size, n_iters, print_counter, inp_path, out_path)

    if testing == True:
        print("Testing...")
        if avai == False:
            print("Genearting new data")
            samples = gg.generate_sample(n_samples) # yields
            np.save(sample_file, samples)
        else:
            samples = np.load(sample_file)
            # Full - dataset
            # data = np.load('../dataset/1200D/1200D_train.npy')
            # data = data.item()
            # samples = data['images']

        tf.reset_default_graph() # important

        # Count #modes
        means = np.load(ds_folder + 'dist_means.npy')
        y_pred = []
        for s in samples:
            pred = classify_dist(s, means, 100)
            if pred > -1:
                y_pred.append(pred)
        x = np.unique(y_pred)
        print("{} modes in {} samples".format(len(x), n_samples))

        # Calculate KL
        pk = get_dist(y_pred, 10)
        ydist = np.load(ds_folder + 'dist_ydist.npy')
        qk = ydist/np.sum(ydist)
        print("qk = ", qk)
        print("pk = ", pk)
        kl_score = entropy(pk, qk)
        print("#KL-score = {:.3}\n".format(kl_score))
