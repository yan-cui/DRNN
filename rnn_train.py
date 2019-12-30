import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
import numpy as np
import scipy.io as sio
import h5py
import os
import time
import math
import sys

print(len(sys.argv))
print([sys.argv[i] for i in range(len(sys.argv))])
if len(sys.argv) < 11:
    print('Usage: python rnn_train.py DATASET[Q1/Q3] TASK MAXSUB STARTSUB HIDDENSIZE NLAYERS CELL[LSTM/GRU/BASIC] EPOCH SHUFFLE[0/1] MARK')
    exit()

# model parameters
DATASET = sys.argv[1]
TASK = sys.argv[2]
MAXSUB = int(sys.argv[3])
SUBIDX = int(sys.argv[4])
HIDDENSIZE = int(sys.argv[5])
NLAYERS = int(sys.argv[6])
RNNCELL = sys.argv[7] # LSTM/GRU/BASIC
MAXEPOCH = int(sys.argv[8])
LRDEPOCH = MAXEPOCH//3.5
learning_rate = 0.004
LRDRATIO = 4 # learning_rate / LRDRATIO
SHUFFLE = int(sys.argv[9])
MARK = sys.argv[10]
dropout_pkeep = 1.0
if DATASET == 'Q1':
    SIGSIZE = 223945 # For HCP68 Q1 data, this size is fixed
elif DATASET == 'Q3':
    if TASK == 'MOTOR':
        SIGSIZE = 244341 # For HCP68 tfMRI Q3 MOTOR data, this size is fixed
    elif TASK == "SOCIAL":
        SIGSIZE = 244329 # For HCP68 tfMRI Q3 SOCIAL data, this size is fixed
BATCHSIZE = 1 # For now batch_size is fixed to 1

STITYPE = 'taskdesign' # 'init_D' for another option
NORM = True # normalization

FASTVALID = False # take init_D as brain signals for fast loading to check code -- unused now
TRAINING = True
VALID = False # unuesd now
TEST= True
HOTTRAIN = False # train on a save checkpoint
TRAINCKPT = '' # if HOTTRAIN is set, TRAINCKPT must be set for training
PLAYCKPT = '' # if TRAINING is not set, PLAYCKPT must be set for test
EXTRAMARK = '' # extra mark for test

SAVECKPT = True # save a checkpoint
SAVESMM = True # save summaries
if FASTVALID:
    SAVECKPT = False # no saving when doing fast valid
    SAVESMM = False
    SIGSIZE = 1000

# task related parameters
if TASK == 'MOTOR':
    SEQLEN = 284
    STISIZE = 6
elif TASK == 'WM':
    SEQLEN = 405
    STISIZE = 4
elif TASK == 'EMOTION':
    SEQLEN = 176
    STISIZE = 2
elif TASK == 'LANGUAGE':
    SEQLEN = 316
    STISIZE = 2
elif TASK == 'GAMBLING':
    SEQLEN = 253
    STISIZE = 2
elif TASK == 'RELATIONAL':
    SEQLEN = 232
    STISIZE = 2
elif TASK == 'SOCIAL':
    SEQLEN = 274
    STISIZE = 2
else:
    print('!!!!!! Error: invalid TASK:', TASK)
    exit()
TRAINLEN = SEQLEN

# data path
DATA_BASE = '/home/cubic/cuiyan/data/HCP/'
STI_DIR = 'HCP_Init_D/'
STI_PREFIX = 'InitD_'+TASK+'_stimulus_hrf_Dsize_1000_sub'
DESIGN_DIR = 'HCP_Taskdesign_allcontrast/'
DESIGN_PREFIX = TASK + '_taskdesign_'
SIG_DIR = 'Whole_b_signals_std_uniform/'+TASK+'/'
if DATASET == 'Q1':
    SIG_DIR_H5 = 'Whole_b_signals_std_uniform_h5norm/'+TASK+'/'
elif DATASET == 'Q3':
    SIG_DIR_H5 = 'Q3_Whole_b_signals_std_uniform_h5norm/'+TASK+'/'
SIG_SUFFIX = '_'+TASK+'_whole_b_signals'

def gen_exp_mark(dataset, task, sbj_idx, max_sub, sti_type, norm, rnn_cell, hidden_size, nlayers, pkeep, mark, shuffle):
    exp_mark = dataset + '-'
    exp_mark = exp_mark + task + '-'
    if max_sub > 1:
        exp_mark = exp_mark + str(max_sub) + 'sub' + str(sbj_idx) + '-' + str(sbj_idx+max_sub-1) + '-'
    else:
        exp_mark = exp_mark + 'sub' + str(sbj_idx) + '-'
    if norm:
        exp_mark = exp_mark + 'N'
    if sti_type == 'init_D':
        exp_mark = exp_mark + 'I'
    elif sti_type == 'taskdesign':
        exp_mark = exp_mark + 'T'
    if shuffle == True:
        exp_mark = exp_mark + "S"
    if rnn_cell == 'LSTM':
        exp_mark = exp_mark + '-L' + str(hidden_size) + 'x' + str(nlayers)
    elif rnn_cell == 'GRU':
        exp_mark = exp_mark + '-G' + str(hidden_size) + 'x' + str(nlayers)
    elif rnn_cell == 'BASIC':
        exp_mark = exp_mark + '-R' + str(hidden_size) + 'x' + str(nlayers)
    if mark != '':
        exp_mark = exp_mark + '-' + mark
    if pkeep != 1.0:
        exp_mark = exp_mark + '-pk' + str(pkeep)
    #exp_mark = exp_mark + '-'
    return exp_mark
    
def normalization(x):
    u = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    m, n = x.shape
    x0 = np.zeros([m,n])
    for i in range(n):
        x0[:,i] = (x[:,i] - u[i]) / sigma[i]
    return x0

# return a file list sorted by subject's index
def list_training_data_h5(sbj_idx, max_sub):
    flist = os.listdir(DATA_BASE + SIG_DIR_H5)
    flist.sort(key = lambda x:int(x[:-(20+len(TASK))]))
    print('Total subjects count:', len(flist), 'requested:', max_sub)

    flist = flist[sbj_idx - 1 : sbj_idx - 1 + max_sub]
    #for i in range(len(flist)):
    #    print('[' + str(i+1) +  ']', flist[i])
    return flist

# load stimuli and hdf5 format whole brain signals
def load_training_data_h5(flist, idx, seq_len, sti_size, sig_size, fast_valid = False, sti = 'init_D', norm = True):
    stimulus = np.zeros([1, seq_len, sti_size])
    signals = np.zeros([1, seq_len, sig_size])

    i = flist[idx][0:len(flist[idx])-20-len(TASK)]
    #print('['+ str(idx+1) +'] Loading training data subject ' + i + '...')
    if sti == 'init_D':
        x = np.loadtxt(DATA_BASE + STI_DIR + STI_PREFIX + i + '.txt')
    elif sti == 'taskdesign':
        #x = sio.loadmat(DATA_BASE + DESIGN_DIR + DESIGN_PREFIX + i + '.mat')['Normalized_'+TASK+'_taskdesign']
        x = sio.loadmat(DATA_BASE + DESIGN_DIR + DESIGN_PREFIX + '1.mat')['Normalized_'+TASK+'_taskdesign']
    else:
        x = np.zeros([seq_len, sti_size])
        print('Error: invalid stimulus type: ' + sti)
    x = x[:,0:sti_size]
    stimulus[0,:,:] = x
    if fast_valid:
        y_ = np.loadtxt(DATA_BASE + STI_DIR + STI_PREFIX + i + '.txt')
    else:
        f = h5py.File(DATA_BASE + SIG_DIR_H5 + i + SIG_SUFFIX + '.h5', 'r')
        y_ = f['data'][:]
        f.close()
    #if norm:
    #    y_ = normalization(y_) # for hdf5 files, normalization is done
    signals[0,:,:] = y_
    return stimulus, signals

# load stimuli and txt format whole brain signals -- slower
def load_training_data(sbj_idx, max_sub, seq_len, sti_size, sig_size, fast_valid = False, sti = 'init_D', norm = True):
    stimulus = np.zeros([max_sub, seq_len, sti_size])
    signals = np.zeros([max_sub, seq_len, sig_size])

    flist = os.listdir(DATA_BASE + SIG_DIR)
    flist.sort(key = lambda x:int(x[:-(21+len(TASK))]))
    #print(flist)
    print('Total subjects count:', len(flist), 'requested:', max_sub)

    for idx in range (max_sub):
        i = flist[idx + sbj_idx - 1][0:len(flist[idx + sbj_idx - 1])-21-len(TASK)]
        print('['+ str(idx+1) +'] Loading training data subject ' + i + '...')
        if sti == 'init_D':
            x = np.loadtxt(DATA_BASE + STI_DIR + STI_PREFIX + i + '.txt')
        elif sti == 'taskdesign':
            x = sio.loadmat(DATA_BASE + DESIGN_DIR + DESIGN_PREFIX + i + '.mat')['Normalized_'+TASK+'_taskdesign']
        else:
            x = np.zeros([seq_len, sti_size])
            print('Error: invalid stimulus type: ' + sti)
        x = x[:,0:sti_size]
        stimulus[idx,:,:] = x
        if fast_valid:
            y_ = np.loadtxt(DATA_BASE + STI_DIR + STI_PREFIX + i + '.txt')
        else:
            y_ = np.loadtxt(DATA_BASE + SIG_DIR + i + SIG_SUFFIX + '.txt')
        if norm:
            y_ = normalization(y_)
        signals[idx,:,:] = y_
    return stimulus, signals

# load test stimulus by keeping the (test_idx)th stimulus and setting others to zeros
def load_play_data(test_idx, seq_len, sti_size, sti_idx, sti = 'init_D'):
    i = test_idx
    test_stimulus = np.zeros([1, seq_len, sti_size])

    #print('Loading test stimulus subject ' + str(i) + '...')
    if sti == 'init_D':
        xin = np.loadtxt(DATA_BASE + STI_DIR + STI_PREFIX + str(i) + '.txt')
    elif sti == 'taskdesign':
        xin = sio.loadmat(DATA_BASE + DESIGN_DIR + DESIGN_PREFIX + str(i) + '.mat')['Normalized_'+TASK+'_taskdesign']
    else:
        xin = np.zeros([seq_len, sti_size])
        print('Error: invalid stimulus type: ' + sti)

    #x = np.zeros([seq_len, sti_size])
    x = -np.ones([seq_len, sti_size])
    if sti_idx > 0 and sti_idx <=sti_size:
        x[:,sti_idx-1:sti_idx] = xin[:,sti_idx-1:sti_idx]
    test_stimulus[0,:,:] = x
    return test_stimulus

# load all test stimulus
def load_full_play_data(test_idx, seq_len, sti_size, sti = 'init_D'):
    i = test_idx
    test_stimulus = np.zeros([1, seq_len, sti_size])

    #print('Loading test stimulus subject ' + str(i) + '...')
    if sti == 'init_D':
        xin = np.loadtxt(DATA_BASE + STI_DIR + STI_PREFIX + str(i) + '.txt')
    elif sti == 'taskdesign':
        xin = sio.loadmat(DATA_BASE + DESIGN_DIR + DESIGN_PREFIX + str(i) + '.mat')['Normalized_'+TASK+'_taskdesign']
    else:
        xin = np.zeros([seq_len, sti_size])
        print('Error: invalid stimulus type: ' + sti)

    x = -np.ones([seq_len, sti_size])
    x[:,0:sti_size] = xin[:,0:sti_size]
    test_stimulus[0,:,:] = x
    return test_stimulus

# Genarate experiment mark
exp_mark = gen_exp_mark(DATASET, TASK, SUBIDX, MAXSUB, STITYPE, NORM, RNNCELL, HIDDENSIZE, NLAYERS, dropout_pkeep, MARK, SHUFFLE)
print('======> ', exp_mark)

# List training subjects
flist = list_training_data_h5(SUBIDX, MAXSUB)

lr = tf.placeholder(tf.float32, name='lr')  # learning rate
pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
batchsize = tf.placeholder(tf.int32, name='batchsize')

# inputs & variables
X = tf.placeholder(tf.float32, [BATCHSIZE, None, STISIZE], name='X')    # [ BATCHSIZE, SEQLEN, STISIZE ]
Y_ = tf.placeholder(tf.float32, [BATCHSIZE, None, SIGSIZE], name='Y_')  # [ BATCHSIZE, SEQLEN, SIGSIZE ]
W = tf.Variable(tf.zeros([HIDDENSIZE, SIGSIZE]))
b = tf.Variable(tf.zeros([SIGSIZE]))

# RNN model
if RNNCELL == 'LSTM':
    cells = [rnn.BasicLSTMCell(HIDDENSIZE, forget_bias=1.0, state_is_tuple=True) for _ in range(NLAYERS)]
elif RNNCELL == 'GRU':
    cells = [rnn.GRUCell(HIDDENSIZE) for _ in range(NLAYERS)]
elif RNNCELL == 'BASIC':
    cells = [rnn.BasicRNNCell(HIDDENSIZE) for _ in range(NLAYERS)]
dropcells = [rnn.DropoutWrapper(cell, input_keep_prob=pkeep) for cell in cells]
multicell = rnn.MultiRNNCell(dropcells, state_is_tuple=True)
zerostate = multicell.zero_state(BATCHSIZE, dtype=tf.float32)
Yr, H = tf.nn.dynamic_rnn(multicell, X, dtype=tf.float32, initial_state=zerostate)
# Yr: [ BATCHSIZE, SEQLEN, HIDDENSIZE]
# H:  [ BATCHSIZE, HIDDENSIZE*NLAYERS ] # this is the last state in the sequence
H = tf.identity(H, name='H')  # just to give it a name
cellweights = multicell.weights
print(cellweights)

# Linear layer to outputs
Yrflat = tf.reshape(Yr, [-1, HIDDENSIZE])    # [ BATCHSIZE x SEQLEN, HIDDENSIZE]
Yflat = tf.matmul(Yrflat, W) + b            # [ BATCHSIZE x SEQLEN, SIGSIZE ]
Yflat_ = tf.reshape(Y_, [-1, SIGSIZE])     # [ BATCHSIZE x SEQLEN, SIGSIZE]

# loss & optimizer
loss = tf.reduce_mean(tf.pow(Yflat - Yflat_, 2))
loss = tf.reshape(loss, [batchsize, -1])      # [ BATCHSIZE, SEQLEN ]
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# status for display
seqloss = tf.reduce_mean(loss, 1)
batchloss = tf.reduce_mean(seqloss)
loss_summary = tf.summary.scalar("batch_loss", batchloss)
lr_summary = tf.summary.scalar('learning_rate', lr)
W_hist = tf.summary.histogram("W", W)
b_hist = tf.summary.histogram("b", b)
summaries = tf.summary.merge([loss_summary, lr_summary, W_hist, b_hist])

# Init for saving models. They will be saved into a directory named 'checkpoints'.
# Only the last checkpoint is kept.
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver(max_to_keep=1)

# init
init = tf.global_variables_initializer()
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.75
#sess = tf.Session(config=config)
sess = tf.Session()

if TRAINING:
    # Init Tensorboard stuff. Logs will be saved into a directiry named 'log'.
    timestamp = str(math.trunc(time.time()))
    if SAVESMM:
        summary_writer = tf.summary.FileWriter("log/" + exp_mark + "-" + timestamp+ "-training", sess.graph)
        if VALID:
            validation_writer = tf.summary.FileWriter("log/" + exp_mark + "-" + timestamp + "-validation")

    # start training
    if HOTTRAIN:
        print('Training on model ', TRAINCKPT)
        saver = tf.train.import_meta_graph('./checkpoints/rnn_train_' + TRAINCKPT + '.meta')
        saver.restore(sess, './checkpoints/rnn_train_' + TRAINCKPT)
    else:
        sess.run(init)

    #Wo, bo, CWo = sess.run([W, b, cellweights])
    #print('W', Wo, ' b', bo)
    #print(len(CWo), [CWo[cidx] for cidx in range(len(CWo))])

    # training loop
    step = 0
    t_tick = time.time()
    sublist = np.arange(1, MAXSUB + 1)
    for epoch in range(1, MAXEPOCH + 1):
        if SHUFFLE:
            np.random.shuffle(sublist)
        print('======> [epoch', epoch, '],subidx list:', sublist)
        epoch_loss = 0;
        for sub in sublist:
            step += 1
            # training on whole batch
            t_start = time.time()
            stimulus, signals = load_training_data_h5(flist, sub-1, SEQLEN, STISIZE, SIGSIZE, FASTVALID, STITYPE)
            t_load = time.time()
            feed_dict = {X: stimulus[:,0:TRAINLEN,:], Y_: signals[:,0:TRAINLEN,:], lr: learning_rate, pkeep: dropout_pkeep, batchsize: BATCHSIZE}
            _, l, bl, smm = sess.run([train_step, seqloss, batchloss, summaries], feed_dict=feed_dict)
            epoch_loss = epoch_loss + bl
            t_run = time.time()
            print(str(step) + ' [epoch ' +  str(epoch) + '] Batchloss ' + str(bl) + ', Timecost ' +
                    str(round(time.time()-t_tick, 3)) + '(' + str(round(t_load-t_start, 3)) + '/' +
                    str(round(t_run-t_load, 3)) + ')')
            t_tick = time.time()
            # save training data into Tensorboard
            if SAVESMM:
                summary_writer.add_summary(smm, step)

            if step % 10 == 0 and VALID:
                feed_dict = {X: stimulus[:,200:,:], Y_: signals[:,200:,:], pkeep: 1.0, batchsize: BATCHSIZE}
                ls, smm = sess.run([batchloss, loss_summary], feed_dict=feed_dict)
                print(str(step) + '=====> validation' + ": Batchloss " + str(ls))
                # save validation data into Tensorboard
                if SAVESMM:
                    validation_writer.add_summary(smm, step)

            # save a checkpoint
            if step % 100 == 1 and SAVECKPT: # mod(step)==1 to forbid overwrite run over ckpt from auto save ckpt
                saver.save(sess, 'checkpoints/rnn_train_' + exp_mark, global_step=step)

        epoch_loss = epoch_loss / MAXSUB
        summary = tf.Summary()
        summary.value.add(tag='epoch_loss', simple_value = epoch_loss)
        summary_writer.add_summary(summary, epoch)

        # learning rate decay
        if epoch % LRDEPOCH == 0:
            learning_rate = learning_rate/LRDRATIO
            print("learning rate decay: ", learning_rate)
    # End of training, save checkpoint
    if SAVECKPT:
        saver.save(sess, 'checkpoints/rnn_train_' + exp_mark, global_step=step)

# Test
if TEST:
    if not TRAINING:
        saver = tf.train.import_meta_graph('./checkpoints/rnn_train_' + PLAYCKPT + '.meta')
        saver.restore(sess, './checkpoints/rnn_train_' + PLAYCKPT)
    else:
        PLAYCKPT = exp_mark + '-' + str(step)

    # Test outputs will be saved into a directory named 'test'.
    if not os.path.exists("test"):
        os.mkdir("test")
    save_path = './test/' + PLAYCKPT + EXTRAMARK
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Test for each stimulus
    for idx in range(0, STISIZE + 1):
        print('======> Testing stimulus curve: ', idx)
        test_stimulus = load_play_data(1, SEQLEN, STISIZE, idx, STITYPE) # take sub 1's as test stimulus
        feed_dict = {X: test_stimulus[:,:,:], pkeep: 1.0, batchsize: BATCHSIZE}
        yo, yro = sess.run([Yflat, Yrflat], feed_dict=feed_dict)
        np.savetxt(save_path + '/Yr' + str(idx) + '.txt', yro, fmt='%.9e')

    # Input full stimuli to get reconstructed signals
    print('======> Testing all stimulus curves')
    test_stimulus = load_full_play_data(1, SEQLEN, STISIZE, STITYPE)
    feed_dict = {X: test_stimulus[:,:,:], pkeep: 1.0, batchsize: BATCHSIZE}
    yo, yro = sess.run([Yflat, Yrflat], feed_dict=feed_dict)
    np.savetxt(save_path + '/Ya.txt', yo, fmt='%.9e')

    # Save parameters of FC layer and RNN layers
    Wo, bo, CWo = sess.run([W, b, cellweights])
    print('======> Saving W and b...')
    print('W', Wo.shape, ' b', bo.shape)
    np.savetxt(save_path + '/W.txt', Wo, fmt='%.9e')
    np.savetxt(save_path + '/b.txt', bo, fmt='%.9e')
    print('======> Saving cell weights...')
    print(len(CWo), [CWo[cidx].shape for cidx in range(len(CWo))])
    if RNNCELL == 'LSTM':
        for cidx in range (len(CWo)//2):
            print('cell', cidx, CWo[cidx*2].shape, CWo[cidx*2+1].shape)
            np.savetxt(save_path + '/cell' + str(cidx) + '_W.txt', CWo[cidx*2])
            np.savetxt(save_path + '/cell' + str(cidx) + '_b.txt', CWo[cidx*2+1])
    elif RNNCELL == 'GRU':
        print('To be added for GRU cell weights')
        # TODO
    elif RNNCELL == 'BASIC':
        print('To be added for BASIC RNN cell weights')
        # TODO


