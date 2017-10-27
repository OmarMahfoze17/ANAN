classdef NeuralNetworks < handle
    properties
        Value
        noLayers
        noTrainPoints
        learnRate
        J
        noUnits
        a; % Each element is an "activation" of unit i in the layer l a[l][i]
        error
        grad
        theta; % matrix of weights controlling the function mapping form layer j to j+1. theta[l][i in layer l+1][i in layer l];
    end
    methods
        %%% ################ Constructor #####################
        function obj=NeuralNetworks(nL,nU,weightsScal)
            obj.noLayers=nL;
            obj.noUnits=nU;
            for l=1:obj.noLayers
                obj.a{l}=zeros(obj.noUnits(l),1);
                obj.error{l}=zeros(obj.noUnits(l),1);
            end
            obj.initializeWeights(-weightsScal,weightsScal);
        end
        
        function Test(obj)
            for l=1:obj.noLayers-1
                size(obj.theta{l})
                size(obj.grad{l})
            end
        end
        %%%####################################################
        %%%################# Initialize Weights ####################
        function initializeWeights(obj,minimum,maximum)
            for l=1:obj.noLayers-1
                obj.theta{l}=minimum + (maximum-minimum)*rand(obj.noUnits(l),obj.noUnits(l+1));
                obj.grad{l}=zeros(obj.noUnits(l),obj.noUnits(l+1));
                size(obj.theta{l});
            end
        end
        %%%####################################################
        %%%#################### Feed Forward #########################
        function feedForward(obj,input)
            obj.a{1}=input;
            for l=2:obj.noLayers
                obj.a{l}=obj.a{l-1}*obj.theta{l-1};
                if l~=obj.noLayers
                    obj.a{l}(1)=1;
                end
            end
        end
        %%%####################################################
        %%%#################### Error #########################
        function Error(obj,target)
            obj.error{obj.noLayers}=[obj.a{obj.noLayers}-target]';
            for l=obj.noLayers-1:-1:1
                obj.error{l}=obj.theta{l}*obj.error{l+1};
            end
        end
        %%%####################################################
        %%%#################### GradiantDescent #########################
        function gradiantDescent(obj,inputs,targets,learnRate)
            J=0;
            for l=1: obj.noLayers-1
                obj.grad{l}=obj.grad{l}*0;
            end
            obj.grad{l}(2);
            for i=1:length(inputs)
                obj.feedForward(inputs(i,:));
                obj.Error(targets(i,:));
                J=J+sum(obj.error{obj.noLayers}.^2);
                for l=1:obj.noLayers-1
                    obj.grad{l}=obj.grad{l}+obj.a{l}'*obj.error{l+1}';
                end
            end
            for l=1:obj.noLayers-1
                obj.grad{l}=obj.grad{l}/length(inputs);
            end
%             obj.grad{l}(2)
            J=J/length(inputs)
            obj.J=J;
            %%%---------------- Correction of weights
            for l=obj.noLayers-1:-1:1
                obj.theta{l}=obj.theta{l}-obj.grad{l}*learnRate;
            end
            
        end
        %%%####################################################
        %%%#################### Train #########################
        function train (obj,inputs,targets,noIterations,learnRate)
            size(targets);
            figure(1)
            hold on
            for i=1:noIterations
                %                 obj.feedForward(inputs(1,:));
                %                 obj.Error(targets(1,:))
                obj.gradiantDescent(inputs,targets,learnRate);
                plot(i,obj.J,'black.')
                drawnow
            end
        end
        %%%####################################################
    end
end