function f=activationFun(z,type,Derivate)

if Derivate=="noDerivate"
    if type=="linear"
        f=z;
    elseif type=="sigmoid"
        f=1./(1+exp(-z));
    elseif type =="tanh"
        f=tanh(z);
    else
        error('OMAR == Activation function not defined')
    end
elseif Derivate=="derivate"
    if type=="linear"
        
        f=1;
    elseif type=="sigmoid"
        a=activationFun(z,type,"noDerivate");
        f=a.*(1-a);
    elseif type =="tanh"
        a=activationFun(z,type,"noDerivate");
        f=1-a.^2;
    else
        error('OMAR == Activation function not defined')
    end
else
    error('OMAR == derivation is not confirmed')
    
end
end