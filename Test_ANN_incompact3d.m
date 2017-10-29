clc
clear all
close all

noUnits=[172 100 10 3]; % number of units for each layer including BAIS unit exept in the output layer which has no BAIS 
noTrainingPoints=3300;
jump=1
noTestPoints=500;
learningRate=0.5;
scal=1.0;

load 'DATA_38500_NZ19';
% load 'DATA_38500_NZ19'
%%%% -------------- Normalize data ------------
% data=data./max(data);
% %%%------------------------------------------------
% inputTrain=[ones(noTrainingPoints,1),inputDataStep_rand(1:noTrainingPoints,:)];
% targetTrain=targetDataStep_rand(1:noTrainingPoints,:);
% 
% inputTest=[ones(noTestPoints,1),inputDataStep_rand(noTrainingPoints+1:noTrainingPoints+noTestPoints,:)];
% targetTest=targetDataStep_rand(noTrainingPoints+1:noTrainingPoints+noTestPoints,:);
%%%------------------------------------------------
inputTrain=[ones(noTrainingPoints,1),inputData(1:noTrainingPoints,:)];
targetTrain=targetData(1:noTrainingPoints,:);

inputTest=[ones(noTestPoints,1),inputData(noTrainingPoints+1:noTrainingPoints+noTestPoints,:)];
targetTest=targetData(noTrainingPoints+1:noTrainingPoints+noTestPoints,:);

ann=NeuralNetworks(length(noUnits),noUnits,scal,'tanh');
load ('WEIGHTS') %XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
ann.theta=weights;    %XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

ann.train(inputTrain(1:jump:end,:),targetTrain(1:jump:end,:),2,learningRate)
%%
% weights=ann.theta;
% save ('WEIGHTS','weights')

ann.test(inputTrain,targetTrain,'noPlot')
costFunTest=ann.costFunTest
figure (3)
plot (targetTrain(:,2),'r')
hold on
plot(ann.predictedOutput(2,:),'black')

% E=targetTest-ann.predictedOutput';
% 
% mean (E.^2);
