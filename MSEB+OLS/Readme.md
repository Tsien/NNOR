### MSEB + ICLS

The target of this experiment is to combine MSEB with ICLS.

Based on NewConvexFunc, add an additional output layer. The parameters of this layer are analytically determined.

This folder contains:

- nnopt2.m   : main file for the training process.
- nnsimul.m  : file to generate the outputs of the artificial neural network.

- logsig_m.m : logsig function.
- ilogsig.m  : inverse of the logsig function.
- dlogsig.m  : derivative of the logsig function.