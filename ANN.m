clc
clear all
close all

noUnits=[4 4 2] % number of units for each layer including BAIS unit exept in the output layer which has no BAIS 
noTrainingPoints=800;
noTestPoints=150;
learningRate=0.15;
scal=1.0;
% data=load('./engine_dataset.txt')
% inputData=data(:,1:noUnits(1));
% outputData=data(:,1:noUnits(end));
[inputData outputData]=engine_dataset;
data=[ones(length(inputData),1) inputData' outputData'];
%%%% -------------- Normalize data ------------
data=data./max(data);
%%%------------------------------------------------
inputTrain=data(1:noTrainingPoints,1:noUnits(1));
targetTrain=data(1:noTrainingPoints,1:noUnits(end));
inputTest=data(noTrainingPoints+1:noTrainingPoints+noTestPoints,1:noUnits(1));
targetTest=data(noTrainingPoints+1:noTrainingPoints+noTestPoints,1:noUnits(end));




ann=NeuralNetworks(length(noUnits),noUnits,1.)
% ann.initializeWeights(-5,5)
ann.train(inputTrain,targetTrain,500,0.1)

