classdef NeuralNetworks < handle
    properties
        Value
        noLayers
        noTrainPoints
        activationFunType
        learnRate
        
        costFunTrain
        costFunTest
        noUnits
        a; % Each element is an "activation" of unit i in the layer l a[l][i]
        z;
        predictedOutput;
        error
        grad
        theta; % matrix of weights controlling the function mapping form layer j to j+1. theta[l][i in layer l+1][i in layer l];
    end
    methods
        %%% ################ Constructor #####################
        function obj=NeuralNetworks(nL,nU,weightsScal,activationFunType)
            
            obj.noLayers=nL;
            obj.noUnits=nU;
            obj.activationFunType=activationFunType;
            for l=1:obj.noLayers
                obj.a{l}=zeros(1,obj.noUnits(l));
                obj.z{l}=zeros(1,obj.noUnits(l));
                obj.error{l}=zeros(obj.noUnits(l),1);
            end
            obj.initializeWeights(-weightsScal,weightsScal);
        end
        
        %         function Test(obj)
        %             for l=1:obj.noLayers-1
        %                 size(obj.theta{l})
        %                 size(obj.grad{l})
        %             end
        %         end
        %%%####################################################
        %%%################# Initialize Weights ####################
        function initializeWeights(obj,minimum,maximum)
            for l=1:obj.noLayers-1
                obj.theta{l}=minimum + (maximum-minimum)*rand(obj.noUnits(l),obj.noUnits(l+1));
                %                 obj.theta{l}=ones(obj.noUnits(l),obj.noUnits(l+1))*.2; %%%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
                obj.grad{l}=zeros(obj.noUnits(l),obj.noUnits(l+1));
                size(obj.theta{l});
            end
        end
        %%%####################################################
        %%%#################### Feed Forward #########################
        function feedForward(obj,input)
            obj.a{1}=input;
            for l=2:obj.noLayers
                obj.z{l}=obj.a{l-1}*obj.theta{l-1};
                obj.a{l}=activationFun(obj.z{l},obj.activationFunType,'noDerivate');
                if l~=obj.noLayers
                    obj.a{l}(1)=1;
                end
            end
        end
        %%%####################################################
        %%%#################### Error #########################
        function Error(obj,target)
            obj.error{obj.noLayers}(:)=obj.a{obj.noLayers}-target;
            for l=obj.noLayers-1:-1:2
                obj.error{l}=obj.theta{l}*obj.error{l+1}.*activationFun(obj.z{l},...
                    obj.activationFunType,'derivate')';
            end
        end
        %%%####################################################
        %%%#################### GradiantDescent #########################
        function gradiantDescent(obj,inputs,targets,learnRate)
            
            for l=1: obj.noLayers-1
                obj.grad{l}=obj.grad{l}*0;
            end
            costFunTrain=0;
            for i=1:length(inputs)
                obj.feedForward(inputs(i,:));
                obj.Error(targets(i,:));
                costFunTrain=costFunTrain+sum(obj.error{obj.noLayers}.^2);
                for l=1:obj.noLayers-1
                    
                    %                     obj.grad{l}=obj.grad{l}+[obj.a{l}]'*obj.error{l+1}';
                    obj.grad{l}=obj.grad{l}+obj.a{l}'*obj.error{l+1}';
                end
            end
            for l=1:obj.noLayers-1
                obj.grad{l}=obj.grad{l}/length(inputs);
            end
            %             obj.grad{l}(2)
            costFunTrain=costFunTrain/length(inputs)
            obj.costFunTrain(end+1)=costFunTrain;
            %%%---------------- Correction of weights
            for l=obj.noLayers-1:-1:1
                obj.theta{l}=obj.theta{l}-obj.grad{l}*learnRate;
            end
            
        end
        %%%####################################################
        %%%#################### Train #########################
        function train (obj,inputs,targets,noIterations,learnRate)
            figure(1)
            hold on
            for i=1:noIterations
                %                 obj.feedForward(inputs(1,:));
                %                 obj.Error(targets(1,:))
                obj.gradiantDescent(inputs,targets,learnRate);
                plot(i,obj.costFunTrain(i),'black.')
                drawnow
            end
        end
        %%%####################################################
        
        %%%####################################################
        %%%#################### Test #########################
        function test (obj,inputs,targets,Plot)
            obj.costFunTest=0;
            if Plot=="plot"
                figure(2)
                grid on
                hold on
                k=1; %% output indix for the plot
                %                 ylim([(min(targets(:,k))-0.2*abs(min(targets(:,k)))) max(targets(:,k))*1.1])
                %                 set(gcf,'Units','normalized','OuterPosition',[0 0 1 1])
                
            end
            for i=1:length(inputs)
                obj.feedForward(inputs(i,:));
                obj.predictedOutput(:,i)=obj.a{obj.noLayers};
                E=obj.a{obj.noLayers}-targets(i,:);
                obj.costFunTest=obj.costFunTest+sum([E.^2]);
                %%----------- plot the predicted values and the true ones -
                if Plot=="plot"
                    if i>1
                        plot([i-1,i],[targets(i-1,k),targets(i,k)],'r','linewidth',2)
                        plot([i-1,i],[aOld,obj.a{obj.noLayers}(k)],'black','linewidth',4)
                        %                         if i==2
                        %                              legend('True Value','Predicted')
                        %                         end
                        if i>15
                            xlim([i-15 i+5])
                        end
                        drawnow;
                        frame(i) = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
                        
                    end
                    aOld=obj.a{obj.noLayers}(1);
                end
            end
            obj.costFunTest=obj.costFunTest/length(inputs);
            
            if Plot=="plot"
                video=VideoWriter('ANN_Test2.avi','Uncompressed AVI');
                video.FrameRate = 10; % How many frames per second.
                open(video);
                
                for i=2:length(inputs)
                    writeVideo(video,frame(i));
                end
                close(video)
            end
            
        end
        %%%####################################################
    end
end