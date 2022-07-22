# GRU implementations using PyTorch for Time Sequence Prediction
This is a toy example for learning both GRU and time sequence prediction. Two GRUCell units are used in this example to learn some sine wave signals starting at different phases. After learning the sine waves, the network tries to predict the signal values in the future. We compare our GRU implementation with pytorch GRU and shown results in the picture below.


This repo contains implementations of:

  * Basic GRUCell
  * Time Sequence Prediction
  
To do:
* Basic RNNCell
* Basic LSTMCell 


## Usage

* using PyTorch GRU Implementation
```
python train.py --gru="torch"
```

* using our GRU Implementation
```
python train.py --gru="custom"
```

## Results
The initial signal and the predicted results are shown in the image. We first give some initial signals (full line). The network will  subsequently give some predicted results (dash line). It can be concluded that the network can generate new sine waves.

* PyTorch GRU:
![image](pics/predict14_gru.png)

* Our GRU Implementation
![image](pics/predict14_our_gru.png)


## Example of LSTM Cell and GRU Cell
![RNN_LSTM_GRU_PyTorch](pics/lstm_gru.png)
Thanks:
> https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21


## Dependencies
* ```pytorch```
* ```numpy```