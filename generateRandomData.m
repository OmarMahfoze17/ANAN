function [x1,x2,x3,x4,y1,y2]=generateRandomData(n)
x1=rand(n,1);
x2=rand(n,1);
x3=rand(n,1);
x4=rand(n,1);
% y=0.5+tanh(5*x1)+tanh(3*x2)
% y=tanh(.2+.4*x1-.6*x2);
% y=5+5*x1+3*x2
z1=.2+.4*x1-.6*cos(x2)+0.8*x3-0.7*x4;
z2=.1+.3*x1-.5*x2+0.7*x3-0.9*sin(x4);

y1=activationFun(z1,'tanh','noDerivate');
y2=activationFun(z2,'sigmoid','noDerivate');
y1=activationFun(y2,'tanh','noDerivate');
y2=activationFun(y1,'sigmoid','noDerivate');
end

