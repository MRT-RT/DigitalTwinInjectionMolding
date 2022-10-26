clear
clc

%% Load Data and subsample
data1 = load('APRBS_Data_1');
data1 = data1.data(1:50:end,:);

data2 = load('APRBS_Data_2');
data2 = data2.data(1:50:end,:);

data3 = load('APRBS_Data_3');
data3 = data3.data(1:50:end,:);

%% Center output data

data1(:,2)=data1(:,2);%-0.107;
data2(:,2)=data2(:,2);%-0.107;
data3(:,2)=data3(:,2);%-0.107;



%% Arange multi-experiment data such that n4sid-Subspace can use it

Ts = 0.01;

train = iddata(data3(:,2),data3(:,1),Ts);
test  = iddata(data2(:,2),data2(:,1),Ts);
%% Identify state space model from data with n4sid

% Overview over options: https://de.mathworks.com/help/ident/ref/n4sidoptions.html
opt = n4sidOptions('InitialState','estimate','N4Weight','auto','Focus','simulation',...
    'WeightingFilter',[],'EnforceStability',0,...
    'Display','on');
    
[ssm,x0] = n4sid(train,3,opt,'DisturbanceModel','none');


%% Simulate identified model
opt = simOptions('InitialCondition',x0);
[y_train,~,x_train] = sim(ssm,train.u,opt);

[y_test,~,x_test] = sim(ssm,test.u);

close all

figure;
hold on
plot(data3(:,2))
plot(y_train(:,1))
hold off

figure;
hold on
plot(data2(:,2))
plot(y_train(:,1))
hold off
%% Save results to python dictionary

Results = struct(...
'A',ssm.A,...
'B',ssm.B,...
'C',ssm.C,...
'D',ssm.D,...
'hidden_train_0',x0,...
'input_train_data',train.u,...
'target_train_data',train.y);

save('Bioreactor_LSS_3states.mat','Results')
