function [MAE, MZOE, W] = myELM(data, NumberofHiddenNeurons)

% Usage: myELM(Data_File, NumberofHiddenNeurons)
% OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm(Data_File, NumberofHiddenNeurons)
%
% Input:
% data                  - dataset
% NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
%
% Output: 
% MAE           - Training and testing MAE
% MZOE          - Training and testing MZOE
% W             - Weights of ELM

%%%%%%%%%%% Load training dataset, the first column is label
T=data(:,end)';% the label
P=data(:,1:end - 1)';% the samples
clear data;                                   %   Release raw training data array

NumberofTrainingData=size(P,2);
NumberofInputNeurons=size(P,1);


%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*P;
clear P;                                            %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H
%%%%%%%% Sigmoid 
H = 1 ./ (1 + exp(-tempH));
clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
OutputWeight=pinv(H') * T';                        % implementation without regularization factor //refer to 2006 Neurocomputing paper

W{1} = InputWeight;
W{2} = OutputWeight;
W{3} = BiasofHiddenNeurons;
%%%%%%%%%%% Calculate error
Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data

clear H;


MAE = sum(abs(T - Y)) / NumberofTrainingData;
Y = ceil(Y - 0.5);
MZOE = numel(find(T ~= Y)) / NumberofTrainingData;

disp(['myELM--> MAE:' num2str(MAE) ', MZOE:' num2str(MZOE)]);
