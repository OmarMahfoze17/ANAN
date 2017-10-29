clc
clear all
close all
Path='/home/omar/PhD/Runs/dataExtraction/uncontroledChannel';
fileNameIntial='ux';
nTime=38500;
dt=0.005;
% ========== Main Grid Size =================
nx=128;
ny=129;
nz=84;
Lx=4*pi;
Ly=2;
Lz=4/3*pi;
dx=Lx/(nx-1);
dz=Lz/(nz-1);
y=load([Path,'/','yp.dat']);
dy=y(2)-y(1);
%#############################################
%############## Saving Box Size ##############
%50,74,1,34,33,48
% 50,74,1,34,47,80
nxB1=50;
nxB2=74;
nyB1=1;
nyB2=34;
nzB1=47;
nzB2=80;
nxBox=nxB2-nxB1+1;
nyBox=nyB2-nyB1+1;
nzBox=nzB2-nzB1+1;
%#############################################
%############## Measuring points ##############
xm1=[12];
ym1=[1];
zm1=[8:26];

xm2=[12];
ym2=[17];
zm2=[11];
%#############################################
%##############  Extract Data   ##############
Step=10;
k=0;
for time=1:nTime
    time
    id=num2str(100000+time);
    ID=id(2:end);
    fileName=[Path,'/subSave/ux/ux',ID];
    ux=readBinay(fileName,nxBox,nyBox,nzBox);
    fileName=[Path,'/subSave/uy/uy',ID];
    uy=readBinay(fileName,nxBox,nyBox,nzBox);
    fileName=[Path,'/subSave/uz/uz',ID];
    uz=readBinay(fileName,nxBox,nyBox,nzBox);
    fileName=[Path,'/subSave/pp/pp',ID];
    pp=readBinay(fileName,nxBox,nyBox,nzBox);
    %%==========================================
    uxMeasure1(:)=ux(xm1,ym1+1,zm1);
    uzMeasure1(:)=uz(xm1,ym1+1,zm1);
    ppMeasure1(:)=pp(xm1,ym1,zm1);
    uxMeasure2(:)=ux(xm2,ym2,zm2);
    uyMeasure2(:)=uy(xm2,ym2,zm2);
    uzMeasure2(:)=uz(xm2,ym2,zm2);
    %%==========================================
    dudy=(uxMeasure1-0)/dy;
    dwdy=(uzMeasure1-0)/dy;
    dpdx(:)=(pp(xm1+1,ym1,zm1)-pp(xm1,ym1,zm1))/dx;
    dpdz(:)=(pp(xm1,ym1,zm1+1)-pp(xm1,ym1,zm1))/dz;
    %%==========================================
    if time>1
        
        DdudyDt=(dudy-dudy_old)/dt;
        DdwdyDt=(dwdy-dwdy_old)/dt;
        DppDt=(ppMeasure1-ppMeasure1_old)/dt;
        DdpdxDt=(dpdx-dpdx_old)/dt;
        DdpdzDt=(dpdz-dpdz_old)/dt;
        if mod(time,Step)==0
            k=k+1
            inputDataStep(k,:)=[dudy,dwdy,dpdx,dpdz,DdudyDt,DdwdyDt,DppDt,DdpdxDt,DdpdzDt];
            targetDataStep(k,:)=[uxMeasure2, uyMeasure2, uzMeasure2];
        end
        inputData(time-1,:)=[dudy,dwdy,dpdx,dpdz,DdudyDt,DdwdyDt,DppDt,DdpdxDt,DdpdzDt];
        targetData(time-1,:)=[uxMeasure2, uyMeasure2, uzMeasure2];
    end
    

ppMeasure1_old=ppMeasure1;
dudy_old=dudy;
dwdy_old=dwdy;
dpdx_old=dpdx;
dpdz_old=dpdz;









end
%%
maxInputData=max(abs(inputData));
inputData=inputData./max(abs(inputData));
% targetData=targetData./max(abs(targetData));

inputDataStep=inputDataStep./max(abs(inputData));
% targetDataStep=targetDataStep./max(abs(targetData));

[~,idx] = sort(rand(size(inputDataStep(:,1))));
for i=1:length(idx)
   inputDataStep_rand(i,:)=inputDataStep(idx(i),:);   
   targetDataStep_rand(i,:)=targetDataStep(idx(i),:);  
end
save ('DATA_ETRAC_1')
save ('DATA_38500_NZ19','inputData','targetData','maxInputData')
save ('DATA_38500_STEP_10_RAND_NZ19','inputDataStep_rand','targetDataStep_rand','maxInputData')