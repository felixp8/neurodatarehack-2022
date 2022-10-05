import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

def theta(x):
    return 0.5*(1 + np.sign(x))

def f(x):
    return np.tanh(x)

def df(x):
    return 1/np.cosh(10*np.tanh(x/10))**2  # the tanh prevents oveflow


class RNN:
    '''
    A recurrent neural network.

    Parameters:
    ----------
    n_in, n_rec, n_out : number of input, recurrent, and hidden units.

    h0 : The initial state vector of the RNN.

    tau_m : The network time constant, in units of timesteps.
    '''

    def __init__(self, n_in, n_rec, n_out, h0, tau_m=10):
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.h0 = h0
        self.tau_m = tau_m

        # Initialize weights:
        self.w_in = 0.1*(np.random.rand(n_rec, n_in) - 1)
        self.w_rec = 1.5*np.random.randn(n_rec, n_rec)/n_rec**0.5
        self.w_out = 0.1*(2*np.random.rand(n_out, n_rec) - 1)/n_rec**0.5

        # Random error feedback matrix:
        self.b = np.random.randn(n_rec, n_out)/n_out**0.5


    def run_trial(self, x, y_, eta=[0.1, 0.1, 0.1],
                  learning=None, online_learning=False):
        '''
        Run the RNN for a single trial.

        Parameters:
        -----------
        x : The input to the network. x[t,i] is input from unit i at timestep t.

        y_ : The target RNN output, where y_[t,i] is output i at timestep t.

        eta : A list of 3 learning rates, for w_in, w_rec, and w_out,
            respectively.

        learning : Specify the learning algorithm with one of the following
            strings: 'rtrl', 'bptt', or 'rflo'. If None, run the network without
            learning.

        online_learning : If True (and learning is on), update weights at each
            timestep. If False (and learning is on), update weights only at the
            end of each trial. Online learning cannot be used with BPTT.

        Returns:
        --------
        y : The time-dependent network output. y[t,i] is output i at timestep t.

        h : The time-dependent RNN state vector. h[t,i] is unit i at timestep t.

        u : The inputs to RNN units (feedforward plus recurrent) at each
            timestep.
        '''

        # Boolean shorthands to specify learning algorithm:
        rtrl = learning == 'rtrl'
        bptt = learning == 'bptt'
        rflo = learning == 'rflo'

        [eta3, eta2, eta1] = eta  # learning rates for w_in, w_rec, and w_out
        t_max = np.shape(x)[0]  # number of timesteps

        dw_in, dw_rec, dw_out = 0, 0, 0  # changes to weights

        u = np.zeros((t_max, self.n_rec))  # input (feedforward plus recurrent)
        h = np.zeros((t_max, self.n_rec))  # time-dependent RNN activity vector
        h[0] = self.h0  # initial state
        y = np.zeros((t_max, self.n_out))  # RNN output
        err = np.zeros((t_max, self.n_out))  # readout error

        # If rflo, eligibility traces p and q should have rank 2; if rtrl, rank 3:
        if rtrl:
            p = np.zeros((self.n_rec, self.n_rec, self.n_rec))
            q = np.zeros((self.n_rec, self.n_rec, self.n_in))
        elif rflo:
            p = np.zeros((self.n_rec, self.n_rec))
            q = np.zeros((self.n_rec, self.n_in))
            
        for jj in range(self.n_rec):
            if rtrl:
                q[jj, jj, :] = df(u[0, jj])*x[0,:]/self.tau_m
            elif rflo:
                q[jj, :] = df(u[0, jj])*x[0,:]/self.tau_m

        for tt in range(t_max-1):
            u[tt+1] = np.dot(self.w_rec, h[tt]) + np.dot(self.w_in, x[tt+1])
            h[tt+1] = h[tt] + (-h[tt] + f(u[tt+1]))/self.tau_m
            y[tt+1] = np.dot(self.w_out, h[tt+1])
            err[tt+1] = y_[tt+1] - y[tt+1]  # readout error

            if rflo:
                p = (1-1/self.tau_m)*p
                q = (1-1/self.tau_m)*q
                p += np.outer(df(u[tt+1,:]), h[tt,:])/self.tau_m
                q += np.outer(df(u[tt+1,:]), x[tt,:])/self.tau_m
            elif rtrl:
                p = np.tensordot((1-1/self.tau_m)*np.eye(self.n_rec)
                    + df(u[tt+1])*self.w_rec/self.tau_m, p, axes=1)
                q = np.tensordot((1-1/self.tau_m)*np.eye(self.n_rec)
                    + df(u[tt+1])*self.w_rec/self.tau_m, q, axes=1)
                for jj in range(self.n_rec):
                    p[jj, jj, :] += df(u[tt+1, jj])*h[tt]/self.tau_m
                    q[jj, jj, :] += df(u[tt+1, jj])*x[tt+1]/self.tau_m

            if rflo and online_learning:
                dw_out = eta1/t_max*np.outer(err[tt+1], h[tt+1])
                dw_rec = eta2*np.outer(np.dot(self.b, err[tt+1]),
                                       np.ones(self.n_rec))*p/t_max
                dw_in = eta3*np.outer(np.dot(self.b, err[tt+1]),
                                      np.ones(self.n_in))*q/t_max
            elif rflo and not online_learning:
                dw_out += eta1/t_max*np.outer(err[tt+1], h[tt+1])
                dw_rec += eta2*np.outer(np.dot(self.b, err[tt+1]),
                                        np.ones(self.n_rec))*p/t_max
                dw_in += eta3*np.outer(np.dot(self.b, err[tt+1]),
                                       np.ones(self.n_in))*q/t_max
            elif rtrl and online_learning:
                dw_out = eta1/t_max*np.outer(err[tt+1], h[tt+1])
                dw_rec = eta2/t_max*np.tensordot(
                    np.dot(err[tt+1], self.w_out), p, axes=1)
                dw_in = eta3/t_max*np.tensordot(
                    np.dot(err[tt+1], self.w_out), q, axes=1)
            elif rtrl and not online_learning:
                dw_out += eta1/t_max*np.outer(err[tt+1], h[tt+1])
                dw_rec += eta2/t_max*np.tensordot(
                    np.dot(err[tt+1], self.w_out), p, axes=1)
                dw_in += eta3/t_max*np.tensordot(
                    np.dot(err[tt+1], self.w_out), q, axes=1)

            if online_learning and not bptt:
                self.w_out = self.w_out + dw_out
                self.w_rec = self.w_rec + dw_rec
                self.w_in = self.w_in + dw_in
            
        if bptt:  # backward pass for BPTT
            z = np.zeros((t_max, self.n_rec))
            z[-1] = np.dot((self.w_out).T, err[-1])
            for tt in range(t_max-1, 0, -1):
                z[tt-1] = z[tt]*(1 - 1/self.tau_m)
                z[tt-1] += np.dot((self.w_out).T, err[tt])
                z[tt-1] += np.dot(z[tt]*df(u[tt]), self.w_rec)/self.tau_m

                # Updates for the weights:
                dw_out += eta1*np.outer(err[tt], h[tt])/t_max
                dw_rec += eta2/(t_max*self.tau_m)*np.outer(z[tt]*df(u[tt]),
                                                            h[tt-1])
                dw_in += eta3/(t_max*self.tau_m)*np.outer(z[tt]*df(u[tt]),
                                                           x[tt])

        if not online_learning:  # wait until end of trial to update weights
            self.w_out = self.w_out + dw_out
            self.w_rec = self.w_rec + dw_rec
            self.w_in = self.w_in + dw_in

        return y, h, u
    

    def run_batch(self, xs, ys_, eta=[0.1, 0.1, 0.1], 
                  learning=None, online_learning=False):
        raise NotImplementedError


    def run_session(self, n_trials, x, y_, eta=[0.1, 0.1, 0.1],
                    learning=None, online_learning=False):
        '''
        Run the RNN for a session consisting of many trials.

        Parameters:
        -----------
        n_trials : Number of trials to run the RNN

        x : The time-dependent input to the RNN (same for each trial).

        y_ : The target RNN output (same for each trial).

        eta : A list of 3 learning rates, for w_in, w_rec, and w_out,
            respectively.

        learning : Specify the learning algorithm with one of the following
            strings: 'rtrl', 'bptt', or 'rflo'. If None, run the network without
            learning.

        online_learning : If True (and learning is on), update weights at each
            timestep. If False (and learning is on), update weights only at the
            end of each trial. Online learning cannot be used with BPTT.


        Returns:
        --------
        y : The RNN output.
        
        loss_list : A list with the value of the loss function for each trial.

        readout_alignment : The normalized dot product between the vectorized
            error feedback matrix and the readout matrix, as in Lillicrap et al
            (2016).
        '''

        t_max = np.shape(x)[0]  # number of timesteps
        loss_list = []
        readout_alignment = []

        # Flatten the random feedback matrix to check for feedback alignment:
        bT_flat = np.reshape((self.b).T,
            (np.shape(self.b)[0]*np.shape(self.b)[1]))
        bT_flat = bT_flat/np.linalg.normnorm(bT_flat)

        for ii in range(n_trials):
            y, h, u = self.run_trial(x, y_, eta, learning=learning,
                                     online_learning=online_learning)

            err = y_ - y
            loss = 0.5*np.mean(err**2)
            loss_list.append(loss)

            w_out_flat = np.reshape(self.w_out,
                (np.shape(self.w_out)[0]*np.shape(self.w_out)[1]))
            w_out_flat = w_out_flat/np.linalg.norm(w_out_flat)
            readout_alignment.append(np.dot(bT_flat, w_out_flat))
            print('\r'+str(ii+1)+'/'+str(n_trials)+'\t Err:'+str(loss), end='')

        return y, loss_list, readout_alignment



def train_unbatched(model, train_input, train_output, valid_input, valid_output, 
                    n_epochs=1, learning_rule='bptt', eta=[0.1, 0.1, 0.1], online_learning=False, loss_func='mse', save_freq=1):
    if loss_func == 'mse':
        def loss_func(target, prediction):
            return np.mean(np.power(target - prediction, 2))
    hidden_size = model.n_rec
    epoch_train_loss_log = []
    epoch_val_loss_log = []
    train_loss_log = [] # averaged per epoch
    val_loss_log = [] # averaged per epoch
    train_out = []
    train_states = []
    valid_out = []
    valid_states = []
    for i in range(n_epochs):
        epoch_train_loss = []
        if (i % save_freq == 0):
            epoch_train_output = np.full(train_output.shape, np.nan)
            epoch_train_states = np.full((train_output.shape[0], train_output.shape[1], hidden_size), np.nan)
        for tid, (trial_input, trial_output) in enumerate(zip(train_input, train_output)):
            pred_output, pred_states, pred_inputs = model.run_trial(
                trial_input, trial_output, eta, learning=learning_rule, online_learning=online_learning
            )
            trial_loss = loss_func(trial_output, pred_output)
            epoch_train_loss.append(trial_loss)
            if (i % save_freq == 0):
                epoch_train_output[tid, :, :] = pred_output
                epoch_train_states[tid, :, :] = pred_states
        epoch_val_loss = []
        if (i % save_freq == 0):
            epoch_val_output = np.full(valid_output.shape, np.nan)
            epoch_val_states = np.full((valid_output.shape[0], valid_output.shape[1], hidden_size), np.nan)
        for tid, (trial_input, trial_output) in enumerate(zip(valid_input, valid_output)):
            pred_output, pred_states, pred_inputs = model.run_trial(
                trial_input, trial_output, eta, learning=None, online_learning=False
            )
            trial_loss = loss_func(trial_output, pred_output)
            epoch_val_loss.append(trial_loss)
            if (i % save_freq == 0):
                epoch_val_output[tid, :, :] = pred_output
                epoch_val_states[tid, :, :] = pred_states
        epoch_train_loss_log.append(epoch_train_loss)
        epoch_val_loss_log.append(epoch_val_loss)
        train_loss_log.append(np.mean(epoch_train_loss))
        val_loss_log.append(np.mean(epoch_val_loss))
        if (i % save_freq == 0):
            train_out.append(epoch_train_output)
            train_states.append(epoch_train_states)
            valid_out.append(epoch_val_output)
            valid_states.append(epoch_val_states)
        print(f'{i}: {np.mean(epoch_train_loss)}, {np.mean(epoch_val_loss)}')
    loss_tup = (train_loss_log, val_loss_log, epoch_train_loss_log, epoch_val_loss_log)
    arr_tup = (np.stack(train_out), np.stack(train_states), np.stack(valid_out), np.stack(valid_states))
    return model, loss_tup, arr_tup


with h5py.File('sim_task.h5', 'r') as h5f:
    train_input = h5f['train_input'][()]
    train_output = h5f['train_output'][()]
    valid_input = h5f['valid_input'][()]
    valid_output = h5f['valid_output'][()]
    train_inds = h5f['train_inds'][()]
    valid_inds = h5f['valid_inds'][()]

rnn = RNN(2, 32, 2, np.zeros(32), tau_m=5.)
rnn, loss, output = train_unbatched(
    model=rnn, 
    train_input=train_input, 
    train_output=train_output, 
    valid_input=valid_input,
    valid_output=valid_output,
    n_epochs=250,
    learning_rule='rflo',
    eta=[0.0001, 0.0001, 0.0001],
    online_learning=False,
    loss_func='mse',
    save_freq=1,
)

tl, vl, etl, vtl = loss
to, ts, vo, vs = output

np.savez(
    'rflo_outputs.npz',
    train_output=to,
    train_states=ts,
    valid_output=vo,
    valid_states=vs
)

plt.plot(tl)
plt.savefig('training.png')

import pdb; pdb.set_trace()

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class VanillaRNN(nn.Module):
#     def __init__(self, input_dim, rec_dim, output_dim):
#         super(VanillaRNN, self).__init__()
#         self.rnn = nn.RNN(input_size=input_dim, hidden_size=rec_dim, num_layers=1, nonlinearity='tanh', batch_first=True)
#         self.readout = nn.Linear(rec_dim, output_dim)
    
#     def forward(self, X):
#         states, _ = self.rnn(X)
#         output = self.readout(states)
#         return output, states

# class RNNTrainer:
#     def __init__(self, rnn_params, loss_func='mse', batch_size=1):
#         super(RNNTrainer, self).__init__()
#         self.model = VanillaRNN(**rnn_params)
#         self.get_loss_func(loss_func)
#         self.batch_size = batch_size
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)
    
#     def train(self, train_input, train_output, valid_input, valid_output, n_epochs=1):
#         train_losses = []
#         valid_losses = []
#         for i in range(n_epochs):
#             train_loss = self.train_epoch(train_input, train_output)
#             valid_loss = self.valid_epoch(valid_input, valid_output)
#             train_losses.append(train_loss)
#             valid_losses.append(valid_loss)
#             print(f'{i}: {train_loss}, {valid_loss}')
#         return train_losses, valid_losses
    
#     def train_epoch(self, train_input, train_output):
#         loss_list = []
#         for i in range(0, train_input.shape[0], self.batch_size):
#             batch_input = train_input[i:(i+self.batch_size)]
#             batch_output = train_output[i:(i+self.batch_size)]
#             batch_pred, _ = self.model(batch_input)
#             batch_loss = self.loss_func(batch_output, batch_pred)
#             self.optimizer.zero_grad()
#             batch_loss.backward()
#             self.optimizer.step()
#             loss_list.append(batch_loss.item())
#         loss = np.mean(loss_list)
#         return loss
    
#     def valid_epoch(self, valid_input, valid_output):
#         loss_list = []
#         for i in range(0, valid_input.shape[0], self.batch_size):
#             batch_input = valid_input[i:(i+self.batch_size)]
#             batch_output = valid_output[i:(i+self.batch_size)]
#             batch_pred, _ = self.model(batch_input)
#             batch_loss = self.loss_func(batch_output, batch_pred)
#             loss_list.append(batch_loss.item())
#         loss = np.mean(loss_list)
#         return loss

#     def get_loss_func(self, loss_func):
#         if loss_func == 'mse':
#             self.loss_func = F.mse_loss
#         else:
#             raise NotImplementedError

# train_input = torch.from_numpy(train_input.astype('float32'))
# train_output = torch.from_numpy(train_output.astype('float32'))
# valid_input = torch.from_numpy(valid_input.astype('float32'))
# valid_output = torch.from_numpy(valid_output.astype('float32'))

# rnn_params = {'input_dim': 2, 'rec_dim': 128, 'output_dim': 2}
# trainer = RNNTrainer(rnn_params, batch_size=1)
# tl, vl = trainer.train(train_input, train_output, valid_input, valid_output, n_epochs=500)

# plt.plot(tl)
# plt.savefig('training.png')

# import pdb; pdb.set_trace()