clear
clc

states = 3;
filename = 'LuGre_LSS_s3.mat';

%% Load Data (first column is time, second input, third output)

dataset1 = load('dataset1.mat');
dataset1 = dataset1.dataset1;

dataset2 = load('dataset2.mat');
dataset2 = dataset2.dataset2;

dataset3 = load('dataset3.mat');
dataset3 = dataset3.dataset3;



%% Arange multi-experiment data surch that n4sid-Subspace can use it
Ts = 1/200;

train = iddata(dataset1(:,2),dataset1(:,1),Ts);                                                      % iddata(y,u,Ts)
val = iddata(dataset2(:,2),dataset2(:,1),Ts);    
test = iddata(dataset3(:,2),dataset3(:,1),Ts);   


%% Identify state space model from data with n4sid

% Overview over options: https://de.mathworks.com/help/ident/ref/n4sidoptions.html
opt = n4sidOptions('InitialState','zero','N4Weight','auto','Focus','simulation',...
    'WeightingFilter',[],'EnforceStability',0,...
    'Display','on');
    
[ssm,x0] = n4sid(train,states,opt,'DisturbanceModel','none','form','free');

%% Simulate identified model
opt = simOptions('InitialCondition',x0);
[y_train,~,x_train] = sim(ssm,train.u,opt);

MSE_train = sum((y_train-train.y).^2)/length(y_train)
BFR_train = 1-sum((train.y-y_train).^2)/sum((train.y-mean(train.y)).^2)

figure
hold on
plot(train.y)
plot(y_train)
hold off

[y_val,~,x_val] = sim(ssm,val.u,opt);

MSE_val = sum((y_val-val.y).^2)/length(y_val)
BFR_val = 1-sum((val.y-y_val).^2)/sum((val.y-mean(val.y)).^2)

figure
hold on
plot(val.y)
plot(y_val)
hold off


[y_test,~,x_test] = sim(ssm,test.u,opt);

MSE_test = sum((y_test-test.y).^2)/length(y_test)
BFR_test = 1-sum((test.y-y_test).^2)/sum((test.y-mean(test.y)).^2)

figure
hold on
plot(test.y)
plot(y_test)
hold off

%% Save results to python dictionary

LuGre_LSS = struct(...
'A',ssm.A,...
'B',ssm.B,...
'C',ssm.C,...
'D',ssm.D,...
'hidden_train_0',x0,...
'input_train_data_1',train.u,...
'target_train_data_1',train.y);

save(filename,'LuGre_LSS')

%%
figure
hold on
plot(train.y)
plot(y_train)
hold off



