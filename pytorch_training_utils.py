# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 08:40:17 2018

@author: Pichau
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats
import numpy as np
import pandas as pd
import copy
from sklearn.metrics import roc_auc_score
from os.path import join
from datetime import datetime

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model,
               criterion,
               optimiser,
               X_train,
               y_train,
               X_val=None,
               y_val=None,
               batch_size=32,
               num_epochs=100,
               patience=10,
               verbose=False):
    """
    Trains a pytorch model.
    
    For the particular model, uses the specified optimiser to minimise the
    criterion on the given training data.
    
    - Training is done in batches, with the number of training examples in
    each batch specified by the batch size.
    
    - If no validation data is provided, the model is trained for the specified 
    number of epochs.
    
    - If validation data is provided, the validation criterion is computed after
    every epoch and training is stopped when (if) no improvement to the 
    criterion is observed after a set number of epochs, defined by the patience.
    
    Parameters
    ----------
    
    model: torch.nn.Module
        A pytorch model which inherits from the nn.Module class.
        
    criterion: valid loss function from torch.nn
        The loss between the model predictions and the true labels 
        which the model is trained to minimise.
        e.g. nn.CrossEntropyLoss()
        
    optimiser: valid optimiser from torch.optim
        Optimiser which attempts to minimise the criterion on the training data
        by updating the parameters specified by the optimisers "params" parameter.
        e.g. optim.SGD(model.parameters())
        
    X_train: torch tensor
        Training features with shape (# training examples, # features).
        
    y_train: torch tensor
        Training labels where y_train[i] is the label of X_train[i].
    
    X_val: None or torch tensor
        Validation features with shape (# validation examples, # features).
        default=None
    
    y_val: None or torch tensor
        Validation labels  where y_val[i] is the label of X_val[i].
        default=None
    
    batch_size: integer
        Number of examples in each batch during training and validation.
        default=32
    
    num_epochs: integer
        Maximum number of passes through full training set
        (possibly less if validation data provided and early stopping triggered).
        default=100
    
    patience: integer
        Only used if validation data provided.
        Number of epochs to wait without improvement to the best validation loss
        before training is stopped.
        default=10
    
    verbose: boolean
        Whether or no to print training and validation loss after every epoch.
        default=False
        
    Returns
    -------
    
    model: torch.nn.Module
        Same model as input but with trained parameters.
        If validation data provided, return model corresponding to best validation loss,
        else returns model after final training epoch
    
    history: dictionary
        Dictionary with following fields:
            train_loss: list with training loss after every epoch
            val_loss: list with validation loss after every epoch (only if validation data provided)
    """
    
    # keep history of training loss
    history = {"train_loss": []}
    
    if X_val is not None:
        # keep history of validation loss
        history["val_loss"] = [] 
        
        # make a copy of model weights
        best_model_weights = copy.deepcopy(model.state_dict())
    
    # create vector we will use to shuffle training data at the beginning of every epoch    
    num_train = X_train.size(0)
    i_shuffle = np.random.choice(num_train, num_train, replace=False)     

    # train in epochs
    for epoch in range(num_epochs):
        if verbose:      
            print("Epoch {}/{}".format(epoch + 1, num_epochs))
            
        # train in batches (with shuffling)
        i_shuffle = i_shuffle[np.random.choice(num_train, num_train, replace=False)]
        training_epoch(model,
                      criterion,
                      optimiser,
                      X_train,
                      y_train,
                      batch_size,
                      i_shuffle)
                
        # evaluate model on whole training set in batches
        train_loss = evaluate_model(model,
                                    criterion,
                                    X_train,
                                    y_train, 
                                    batch_size)["loss"]
        history["train_loss"].append(train_loss)
        if verbose:
            print("Train loss: {:.4f}".format(train_loss))
        
        if X_val is not None:
            # evaluate model on whole validation set in batches
            val_loss = evaluate_model(model,
                                     criterion,
                                     X_val,
                                     y_val, 
                                     batch_size)["loss"]
            history["val_loss"].append(val_loss)
            if verbose:
                print("Val loss: {:.4f}".format(val_loss))
        
            # if first epoch or validation loss improved,
            # record best loss and make a copy of best model weights
            if epoch == 0:
                best_loss = val_loss
                epochs_since_improvement = 0
                best_model_weights = copy.deepcopy(model.state_dict())
                
            elif val_loss < best_loss:    
                best_loss = val_loss
                epochs_since_improvement = 0
                best_model_weights = copy.deepcopy(model.state_dict())
                
            else:
                epochs_since_improvement += 1

            # stop training early?
            if epochs_since_improvement == patience:
                # load best model weights and stop training
                model.load_state_dict(best_model_weights)
                return model, history 
            
        if verbose:
            print("-" * 20)
                
    return model, history     


def training_epoch(model,
                  criterion,
                  optimiser,
                  X,
                  y,
                  batch_size=32,
                  i_shuffle=None):
    """
    Executes one training epoch.
    
    For the particular model, updates parameters by performing one pass
    through the training set, using the specified optimiser to minimise
    the specified critierion on the given training data.
    
    - Training is done in batches, with the number of training examples in
    each batch specified by the batch size.
    
    - Training examples can effectively be shuffled by specifying a random order 
    in which the examples are to appear in the batches.
    
    Parameters
    ----------
    
    model: torch.nn.Module
        A pytorch model which inherits from the nn.Module class.
        
    criterion: valid loss function from torch.nn
        The loss between the model predictions and the true labels 
        which the model is trained to minimise.
        e.g. nn.CrossEntropyLoss()
        
    optimiser: valid optimiser from torch.optim
        Optimiser which attempts to minimise the criterion on the training data
        by updating the parameters specified by the optimisers "params" parameter.
        e.g. optim.SGD(model.parameters())
        
    X: torch tensor
        Training features with shape (# training examples, # features).
        
    y: torch tensor
        Training labels where y[i] is the label of X[i].
    
    batch_size: integer
        Number of examples in each batch during training.
        default=32
    
    i_shuffle: None or numpy vector
        If not None, needs to contain indices of training data rows,
        which specifies the order in which the examples are to appear 
        in the batches during training
        default=None
          
    Returns
    -------
    
    model: torch.nn.Module
        Same model as input but with trained parameters.
    """
    
    # we are in training mode
    model.train()
    
    # train in batches
    num_train = X.size(0)
    num_batches = int(np.ceil(num_train / batch_size))    
    for batch in range(num_batches):

        # zero the parameter gradients
        optimiser.zero_grad()

        # get data in this batch
        i_first = batch_size * batch
        i_last = batch_size * (batch + 1)  
        i_last = min(i_last, num_train)        
        if i_shuffle is None:
            X_batch = X[i_first:i_last]
            y_batch = y[i_first:i_last]
        else:
            X_batch = X[i_shuffle[i_first:i_last]]
            y_batch = y[i_shuffle[i_first:i_last]]

        # forward pass
        y_pred = model(X_batch).squeeze()

        # compute loss
        loss = criterion(y_pred, y_batch)

        # backward pass + optimise
        loss.backward()
        optimiser.step()
        
    return model
    

def apply_model(model,
                X,
                batch_size=32):
    """
    Applies a pytorch model in batches.
    
    Uses the given model to predict labels for given features.
    
    - The features must correspond to the same features which were used
    to train the model.
    
    - The model is applied in batches, with the number of examples in
    each batch specified by the batch size.
    
    Parameters
    ----------
    
    model: torch.nn.Module
        A pytorch model which inherits from the nn.Module class.
        
    X: torch tensor
        Features with shape (# examples, # features).
        Must correspond to the same features which were used to train the model.

    batch_size: integer
        Number of examples in each batch during prediction.
        default=32
          
    Returns
    -------
    
    y_pred: pytorch tensor
        Model predictions with shape (# examples,).
    """
    
    # we are in inference mode
    model.eval()
    
    num_points = X.size(0)
    num_batches = int(np.ceil(num_points / batch_size))  
    y_pred = torch.zeros(num_points).to(model.device)
    for batch in range(num_batches):

        # get data in batch
        i_first = batch_size * batch
        i_last = batch_size * (batch + 1)  
        i_last = max(i_last, num_points)                
        X_batch = X[i_first:i_last]

        # predict
        with torch.no_grad():           
            y_pred[i_first:i_last] = model(X_batch).squeeze()

    return y_pred


def evaluate_model(model,
                  criterion, 
                  X,
                  y,
                  batch_size=32,
                  metrics=None):
    """
    Evaluates a pytorch model.
    
    Uses the given model to predict labels for given features,
    and computes overall loss between predictions and given labels
    using the specified criterion. Also computes any additional specified
    metrics.
    
    - The features must correspond to the same features which were used
    to train the model.
    
    - The model is applied in batches, with the number of examples in
    each batch specified by the batch size.
    
    Parameters
    ----------
    
    model: torch.nn.Module
        A pytorch model which inherits from the nn.Module class.
        
    criterion: valid loss function from torch.nn
        The loss between the model predictions and the true labels 
        which the model is trained to minimise.
        e.g. nn.CrossEntropyLoss()
        
    X: torch tensor
        Training features with shape (# examples, # features).
        
    y: torch tensor
        Training labels where y[i] is the label of X[i].
    
    batch_size: integer (default=32)
        Number of examples in each batch during prediction.
        
    metrics: None or list of strings (default=None)
        Metrics to compute between model predictions and true labels.
        Binary classification options are "accuracy" and "auc".
          
    Returns
    -------
    
    scores: dictionary
        Keys are metric names and values are the corresponding scores.
    """
    
    scores = {}
    p = None
    
    with torch.no_grad():
    
        # apply model
        y_pred = apply_model(model, X, batch_size)
        
        # compute the loss
        scores["loss"] = criterion(y_pred, y).item()
        
        if metrics is not None:
            # compute other metrics
            for metric in metrics:  
                
                if metric == "accuracy":
                    if len(y.size()) == 1:
                        p = F.sigmoid(y_pred)
                        scores[metric] = torch.mean((y == (p > 0.5).float()).float()).item()
                    else:
                        pass
                        
                if metric == "auc":
                    if len(y.size()) == 1:
                        if p is None:
                            p = F.sigmoid(y_pred)                    
                        scores[metric] = roc_auc_score(y.cpu().numpy(), p.cpu().numpy())

    return scores


class FeedForwardNet(nn.Module):
    
    def __init__(self,
                input_dim,  
                output_dim,
                params,
                embedding_groups=None,
                device=DEVICE):
        """
        Fully-connected feed forward neural network.
        
        Builds on the standard nn.Module class by adding functionality
        to easily set network hyperparameters and include features such as
        batch normalisation and categorical embeddings.
        
        As well as the initialisation method which constructs the network,
        there are two more methods for initialising layer weights
        and computing the forward pass.
        
        - If embedding groups provided, categorical variables are embedded
        as described in "Entity Embeddings of Categorical Variables" 
        (https://arxiv.org/abs/1604.06737).
        
        Parameters
        ----------
        
        input_dim: integer
            Number of input features.
            
        output_dim: integer
            Number of outputs.
            
        params: dictionary
            Network parameters. 
            See below for available parameters and default values.

        embedding_groups: None or list of lists
            If not None, each list specifies the indices of dummy columns
            corresponding to a single categorical variable.
            
        device: torch device (default=DEVICE)
            Device on which to run computation (CPU or GPU).          
        """
        
        super(FeedForwardNet, self).__init__()
        
        self.device = device
        self.embedding_groups = embedding_groups
        
        # define default network parameters
        self.params = {}
        self.params["num_hidden"] = 1
        self.params["hidden_dim"] = input_dim
        self.params["embed_dummies"] = False
        self.params["embedding_dim"] = 1
        self.params["batch_norm"] = True
        self.params["weight_init"] = "xavier"
        self.params["activation"] = "relu"
   
        # replace provided parameters
        for key, value in params.items():
            if key in self.params:
                self.params[key] = value
        
        # make sure integer parameters are correct type
        input_dim = int(input_dim)
        output_dim = int(output_dim)
        self.params["num_hidden"] = int(self.params["num_hidden"])
        self.params["hidden_dim"] = int(self.params["hidden_dim"])
        
        if embedding_groups is not None:
            # construct embedding for each categorical variable
            self.embeddings = nn.ModuleList()
            total_embedding_dim = 0
            total_dummies = 0
            for dummy_indices in embedding_groups:
                num_dummies = len(dummy_indices)
                self.embeddings.append(nn.Embedding(num_dummies,
                                                   self.params["embedding_dim"]))
                total_dummies += num_dummies
                total_embedding_dim += self.params["embedding_dim"]
                
                self.init_layer(self.embeddings[-1])
            
            # input dimension to first hidden layer is 
            # number of numeric variables + total embedding dimension
            input_dim = input_dim - total_dummies + total_embedding_dim
            
            # save total embedding dimension (we will need it for the forward pass)
            self.total_embedding_dim = total_embedding_dim
        
        # define module lists for storing hidden layers
        self.hidden_layers = nn.ModuleList()
        if self.params["batch_norm"]:
            self.batch_norm_layers = nn.ModuleList()
        
        # construct hidden layers
        for i in range(self.params["num_hidden"]):
            
            if i == 0:  
                self.hidden_layers.append(nn.Linear(input_dim, 
                                                    self.params["hidden_dim"]))
            else:
                self.hidden_layers.append(nn.Linear(self.params["hidden_dim"], 
                                                    self.params["hidden_dim"]))
            
            self.init_layer(self.hidden_layers[-1])
                
            if self.params["batch_norm"]:
                self.batch_norm_layers.append(nn.BatchNorm1d(self.params["hidden_dim"]))
                
        # construct output layer
        self.output_layer = nn.Linear(self.params["hidden_dim"], output_dim)
        self.init_layer(self.output_layer)
        
        
    def init_layer(self, layer):
        """
        Initialises network layer weights.
        
        Initialises the weights of the specified layer using the strategy 
        specified in self.params["weight_init"].
        
        - Available initialisation strategies are "xavier".
        
        - Default is uniform.
        
        Parameters
        ----------
        
        layer: pytorch layer
            Layer whose weights we want to initialise.        
        """
        
        if self.params["weight_init"] == "xavier":
            torch.nn.init.xavier_uniform_(layer.weight)
        else:
            torch.nn.init.uniform_(layer.weight)
            
                
    def forward(self, X):
        """
        Network forward pass.
        
        Computes the network output by performing the forward pass.
        
        - The number of features in the input data must be the same
        as the network input dimension
        
        Parameters
        ----------
        
        X: torch tensor
            Features with shape (# examples, # features).
            
        Returns
        -------
        
        out: torch tensor
            Output of final network layer.            
        """
        
        if self.embedding_groups is not None:
            # create tensor for storing all embeddings
            embeds = torch.zeros((X.size(0), self.total_embedding_dim)).to(self.device)
            
            # loop through each group of dummy variables
            first_col = 0
            last_col = self.params["embedding_dim"]
            all_dummy_indices = []
            for i, dummy_indices in enumerate(self.embedding_groups):
                # get embeddings for this group of dummies
                embed_indices = torch.argmax(X[:, dummy_indices], 1)
                embeds[:, first_col:last_col] = self.embeddings[i](embed_indices)           
                first_col += self.params["embedding_dim"]
                last_col += self.params["embedding_dim"]
                all_dummy_indices += dummy_indices
                
            # concatenate numeric variables with embeddings
            numeric_indices = list(set(range(X.size(1))) - set(all_dummy_indices))
            X = torch.cat((X[:, numeric_indices], embeds), 1)
        
        # loop through hidden layers
        for i, layer in enumerate(self.hidden_layers):
            
            X = layer(X)
            
            # activation
            if self.params["activation"] == "relu":
                X = F.relu(X)
                
            # batch norm
            if self.params["batch_norm"]:
                X = self.batch_norm_layers[i](X)
        
        # output layer
        out = self.output_layer(X)
        
        return out


class NNRandomSearch(object):
    """
    Neural network hyperparameter random search.
    
    Runs random hyperparameter search of neural network architecture and
    training parameters, where values are sampled from pre-defined
    distributions.
    
    Parameters
    ----------
    
    prediction_type: string, one of "binary_classification", "multi_classification" or "regression"
        Specifies the type of task.
        
    param_dists:  dictionary
        The dictionary key specifies the hyperparameter and the value
        specifies the distribution from which to draw samples. 
        The value is a tuple with  two elements.
        The first element defines the distribution, which can be a list
        or a any distribution from scipy.stats.
        The second element is a string which defines how the distribution 
        is sampled. Options are "set" (choose randomly from set),
        "uniform" (sample uniformly) and "exp" (sample uniformly from log
        domain before exponentiating).
        Examples can be found below for default distributions.
        
    criterion: valid loss function from torch.nn (default=None)
        The loss between the model predictions and the true labels 
        which the model is trained to minimise.
        e.g. nn.CrossEntropyLoss().
        
    embedding_groups: None or list of lists (default=None)
        If not None, each list specifies the indices of dummy columns
        corresponding to a single categorical variable.
        
    metrics: None or list of strings (default=None)
        Metrics to compute between model predictions and true labels.
        Binary classification options are "accuracy" and "auc".
        
    results_dir: string (default="")    
        Path to directory for saving results of hyperparameter search.
        
    device: torch device (default=DEVICE)
        Device on which to run computation (CPU or GPU).
    """
        
    def __init__(self,
                prediction_type,
                param_dists,
                criterion=None,
                embedding_groups=None,
                metrics=None,
                results_dir="",
                device=DEVICE):
        """

        """
        
        self.prediction_type = prediction_type
        self.embedding_groups = embedding_groups
        self.metrics = metrics
        self.results_dir = results_dir
        self.device = device
                
        if criterion is None:
            # define loss function
            if self.prediction_type == "binary_classification":                 
                self.criterion = nn.BCEWithLogitsLoss()
            elif self.prediction_type == "multi_classification":
                self.criterion = nn.CrossEntropyLoss()
            else:
                self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion
        
        # define default parameter distributions
        self.param_dists = {}
        self.param_dists["num_hidden"] = ([1], "set")
        self.param_dists["hidden_dim"] = (stats.randint(10, 21), "uniform")
        self.param_dists["embed_dummies"] = ([False], "set")
        self.param_dists["embedding_dim"] = ([1], "set")
        self.param_dists["batch_norm"] = ([True], "set")
        self.param_dists["weight_init"] = (["xavier"], "set")
        self.param_dists["activation"] = (["relu"], "set")
        self.param_dists["optimiser"] = (["adam"], "set")
        self.param_dists["lr"] = (stats.uniform(np.log(0.0001), np.log(0.01) - np.log(0.0001)), "exp")
        
        # replace with provided distributions
        for key, value in param_dists.items():
            if key in self.param_dists:
                self.param_dists[key] = value
        
    
    def fit(self,
           X_train,
           y_train,
           X_val,
           y_val,
           num_experiments,
           num_trials,
           batch_size=32,
           num_epochs=100,
           patience=10,
           verbose=True):
        """
        Runs the random hyperparameter search.
        
        Parameters
        ----------
        
        X_train: torch tensor
            Training features with shape (# training examples, # features).
        
        y_train: torch tensor
            Training labels where y_train[i] is the label of X_train[i].
        
        X_val: torch tensor
            Validation features with shape (# validation examples, # features).
        
        y_val: torch tensor
            Validation labels where y_val[i] is the label of X_val[i].
            
        num_experiments: integer
            Number of experiments to run. 
            Each experiment can contain several trials, and the results of each
            experiment will be saved to a separate file.
            
        num_trials: integer
            Number of neural networks to train in each experiment.
            
        batch_size: integer (default=32)
            Number of examples in each batch during training and validation.       
        
        num_epochs: integer (default=100)
            Maximum number of passes through full training set
            (possibly less if validation data provided and early stopping triggered).
                    
        patience: integer (default=10)
            Only used if validation data provided.
            Number of epochs to wait without improvement to the best validation loss
            before training is stopped.
        
        verbose: boolean (default=True)
            Whether or not to print results of each trial.
        """
        
        # move data to correct device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        
        # get input and output dimensions
        input_dim = X_train.size(1)
        if len(y_train.size()) == 1:
            output_dim = 1
        else:
            output_dim = self.y_train.size(1)
        
        # for each experiment...
        for experiment in range(num_experiments):
            
            # create dataframe for storing results of experiment
            experiment_results = pd.DataFrame()
            
            # for each trial...
            for trial in range(num_trials):
                if verbose:
                    print("Running trial {}/{} of experiment {}/{} ...".format(
                                                                   trial + 1, 
                                                                   num_trials,
                                                                   experiment + 1,
                                                                   num_experiments))                
                
                # sample hyperparameters
                params = self.sample_hyperparameters()                
                trial_results = params.copy()
                
                # build model
                if self.param_dists["embed_dummies"]:
                    embedding_groups = self.embedding_groups
                else:
                    embedding_groups = None                    
                model = FeedForwardNet(input_dim,
                                      output_dim, 
                                      params,
                                      embedding_groups,
                                      self.device)
                model.to(self.device)
                                    
                # define optimiser
                optimiser = self.get_optimiser(model, params)
                
                # train model
                model, history = train_model(model,
                                            self.criterion,
                                            optimiser,
                                            X_train,
                                            y_train,
                                            X_val,
                                            y_val,
                                            batch_size,
                                            num_epochs,
                                            patience,
                                            False)
                
                # get best epoch
                best_epoch = np.argmin(np.array(history["val_loss"])) + 1
                trial_results["epochs"] = best_epoch
                if verbose:
                    print("Best epoch: {}".format(best_epoch))
                
                # compute training metrics
                metric_scores_train = evaluate_model(model,
                                                     self.criterion,
                                                     X_train,
                                                     y_train,
                                                     batch_size,
                                                     self.metrics)
                
                # compute validation metrics
                metric_scores_val = evaluate_model(model,
                                                   self.criterion,
                                                   X_val,
                                                   y_val,
                                                   batch_size,
                                                   self.metrics)
        
                for metric, score_train in metric_scores_train.items(): 
                    score_val = metric_scores_val[metric]
                    trial_results["train_" + metric] = score_train
                    trial_results["val_" + metric] = score_val
                    if verbose:
                        print("Train {}: {:.4f}, Val {}: {:.4f}".format(metric,
                                                                        score_train,
                                                                        metric,
                                                                        score_val,
                                                                        ))
                
                # append trial results
                experiment_results = experiment_results.append(trial_results, ignore_index=True)

                if verbose:
                    print("-" * 50)
                
            # save experiment results
            self.save_results(experiment_results)    
                
    def sample_hyperparameters(self):
        """
        Samples hyperparameters from pre-defined distributions.
        
        Returns
        -------
        
        params: dictionary
            Keys are parameter names and values are parameter values. 
        """
        
        params = {}
        for key, (dist, sampling) in self.param_dists.items():            
            
            if sampling == "uniform":
                params[key] = dist.rvs(size=1)[0]                
                
            if sampling == "exp":
                params[key] = np.exp(dist.rvs(size=1)[0])
                                
            if sampling == "set":
                params[key] = dist[stats.randint(0, len(dist)).rvs(size=1)[0]]
                
        return params
    
    
    def get_optimiser(self,
                     model, 
                     params):
        """
        Defines the optimiser using the specified parameters.
        
        Parameters
        ----------
        
        model: torch.nn.Module
            A pytorch model which inherits from the nn.Module class.
            
        params: dictionary
            Keys are parameter names and values are parameter values. 
        
        Returns
        -------
        
        optimiser: optimiser from torch.optim
            Optimiser defined by given parameters.
        """
        
        if params["optimiser"] == "adam":
            optimiser = optim.Adam(model.parameters(), lr=params["lr"])
            
        else:
            optimiser = optim.SGD(model.parameters(), lr=params["lr"])
            
        return optimiser
    
    
    def save_results(self, 
                     results):
        """
        Saves results of hyperparameter search to disk.
        
        Saves Pandas DataFrame to csv with datetime as filename. 
        
        Parameters
        ----------
        
        results: Pandas DataFrame
            Hyperparameter values and training and validation scores for each trial.
        """
        
        time_now = str(datetime.now())
        time_now = time_now.replace(':', '')
        time_now = time_now.replace('.', '')
        results_file = join(self.results_dir, time_now) + ".csv"
        results.to_csv(results_file, index=False)



 
 
"""---------------------------------------ROUGH WORK BELOW---------------------------------"""
      
def to_sparse(X):
    n, m = X.size()
    rows = torch.range(0, n-1).view(-1, 1).long()
    cols = torch.range(0, m-1).view(1, -1).long()
    rows = rows.repeat(1, m).view(1, -1)
    cols = cols.repeat(1, n)
    idx = torch.cat((rows, cols))
    v = X.view(-1, 1).squeeze()
    
    X_typename = torch.typename(X).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, X_typename)
    
    return sparse_tensortype(idx, v.cpu(), X.size())      
        