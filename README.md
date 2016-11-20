**Name the folders to save logs or checkpoints as run, log, or checkpoint so that git can correctly ignore them**

*Have a look at attentive-reader/utils for all tools*


==============
Attentive Model: Replication of [Teaching machine to read and comprehend](https://arxiv.org/abs/1506.03340) is done. I removed the Tanh layer before cross entropy loss, for nonlinearty here won't help much and deleting it allows gradients flows back better and allows us to use higher learingrate and RMS. The model can overfit trainingset of 3000 samples. I am running test on larger data. 
    My work is partly based on carpedm20's [code](https://github.com/carpedm20/attentive-reader-tensorflow)


=============
Other Model: in construction
