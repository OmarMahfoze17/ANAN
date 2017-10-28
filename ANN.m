clc
clear all
close all

noUnits=[5 2] % number of units for each layer including BAIS unit exept in the output layer which has no BAIS 
noTrainingPoints=800;
noTestPoints=150;
learningRate=0.15;
scal=1.0;
% data=load('./engine_dataset.txt')
% inputData=data(:,1:noUnits(1));
% outputData=data(:,1:noUnits(end));
[inputData outputData]=engine_dataset;
 [x1,x2,x3,x4,y1,y2]=generateRandomData();
% data=[ones(length(inputData),1) inputData' outputData'];
inputs=[ones(length(x1),1),x1,x2,x3,x4];
targets=[y1 y2];
%%%% -------------- Normalize data ------------
% data=data./max(data);
%%%------------------------------------------------
inputTrain=inputs(1:noTrainingPoints,:)
targetTrain=targets(1:noTrainingPoints,:)

inputTest=inputs(noTrainingPoints+1:noTrainingPoints+noTestPoints,:)
targetTest=targets(noTrainingPoints+1:noTrainingPoints+noTestPoints,:)
% 
% targetTrain=data(1:noTrainingPoints,end-noUnits(end)+1:end);
% inputTest=data(noTrainingPoints+1:noTrainingPoints+noTestPoints,1:noUnits(1));
% targetTest=data(noTrainingPoints+1:noTrainingPoints+noTestPoints,1:noUnits(end));




ann=NeuralNetworks(length(noUnits),noUnits,1.,'linear')
% ann.initializeWeights(-5,5)
ann.train(inputTrain,targetTrain,500,.1)
weights=ann.theta{1};
ann.test(inputTrain,targetTrain)

