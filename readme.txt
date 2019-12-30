
rnn_train.py
Train the DRNN model and test.
Usage: python rnn_train.py DATASET[Q1/Q3] TASK MAXSUB STARTSUB HIDDENSIZE NLAYERS CELL[LSTM/GRU/BASIC] EPOCH SHUFFLE[0/1] MARK
After training and test, you'll find the following files under the directory ./test/experiment_mark
W.txt 
b.txt		-- weight and bias of the following layer, W.txt is used for extract spatial patterns.
yr*.txt		-- output of the top rnn layer, a.k.a temporal responses. 
			   '1' ~ 'N' means responses to N test stimuli.
			   '0' means responses to no stimulus input (all zeros).
			   'a' means responses to full stimuli input.
ya.txt		-- reconstructed signals.
cell*_w.txt
cell*_b.txt	-- weights and biases of RNN cells (only support LSTM cells yet).

