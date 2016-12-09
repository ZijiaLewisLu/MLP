Log
==============
**Finished**: Attentive Model from [Teaching machine to read and comprehend](https://arxiv.org/abs/1506.03340).

With batch size=128, vocab size=50000, learning rate 5e-5, RMSProp optimizer, the model achieves 55% accuracy on test set.
    
To train the model (First preprocess the data using tools in utils/data\_utils.py):
```
cd ./attentive-reader
./main.py --learning_rate 0.00005 --vocab_size 50000 --optim RMS --attention concat --activation tanh
```

**Current**:
playing with structures, trying different attention mechanism

My work is partly based on carpedm20's [Code](https://github.com/carpedm20/attentive-reader-tensorflow). 

=============
Other Model: in construction
