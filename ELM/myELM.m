function [Time, Acc, MAE] = myELM(data, NumberofHiddenNeurons)

% Usage: elm(Data_File, NumberofHiddenNeurons)
% OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm(Data_File, NumberofHiddenNeurons)
%
% Input:
% data                  - dataset
% NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
%
% Output: 
% TrainingTime          - Time (seconds) spent on training ELM
% TestingTime           - Time (seconds) spent on predicting ALL testing data
% TrainingAccuracy      - Training accuracy: 
%                           RMSE for regression or correct classification rate for classification
% TestingAccuracy       - Testing accuracy: 
%                           RMSE for regression or correct classification rate for classification
%

[num, dim] = size(data);
index = randperm(num);
test_num = ceil(0.2 * num);
train_data = data(index(test_num + 1 : end), :);
test_data = data(index(1 : test_num), :);

%%%%%%%%%%% Load training dataset, the first column is label
T=train_data(:,end)';% the label
P=train_data(:,1:end - 1)';% the samples
clear train_data;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset
TV.T=test_data(:,end)';
TV.P=test_data(:,1 : end - 1)';
clear test_data;                                    %   Release raw testing data array

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);

%%%%%%%%%%% Calculate weights & biases
start_time_train=cputime;

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

end_time_train=cputime;
Time(1)=end_time_train-start_time_train;        %   Calculate CPU time (seconds) spent for training ELM

%%%%%%%%%%% Calculate the training accuracy
Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data

Acc(1)=sqrt(mse(T - Y));               %   Calculate training accuracy (RMSE) for regression case

clear H;
MAE(1) = sum(abs(T - Y)) / NumberofTrainingData;

%%%%%%%%%%% Calculate the output of testing input
start_time_test=cputime;
tempH_test=InputWeight*TV.P;
clear TV.P;             %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
%%%%%%%% Sigmoid 
H_test = 1 ./ (1 + exp(-tempH_test));
TY=(H_test' * OutputWeight)';                       %   TY: the actual output of the testing data
end_time_test=cputime;
Time(2)=end_time_test-start_time_test;           %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data

Acc(2)=sqrt(mse(TV.T - TY));            %   Calculate testing accuracy (RMSE) for regression case

MAE(2) = sum(abs(TV.T - TY)) / NumberofTestingData;


