clear
clc

%% Load Data
data = load('APRBS_Ident_Data');
data = data.data;

%% Remove Offset Errors in Input and Output as suggested in README.txt



%% Scale Data?


%% Select which data to use for identification, validation and testing

% Identifikationsdaten = [SNLS80mV.V1(1:4E4);SNLS80mV.V2(1:4E4)]';            % White Gaussian noise with increasing amplitude
% 
% Validierungsdaten    = [SNLS80mV.V1(4.05E4+40:4.05E4+8720);...
%     SNLS80mV.V2(4.05E4+40:4.05E4+8720)]';                                   % Odd random multisine
% 
% Testdaten    = [Schroeder80mV.V1(1.055E4:2.19E4);...
%     Schroeder80mV.V2(1.055E4:2.19E4)]';                                     % Multisine with Schroeder phases


%% Save Data Sets for use in python

% save('data/Silverbox/Identifikationsdaten.mat','Identifikationsdaten')
% save('data/Silverbox/Validierungsdaten.mat','Validierungsdaten')
% save('data/Silverbox/Testdaten.mat','Testdaten')



%% Arange multi-experiment data such that n4sid-Subspace can use it

Ts = 0.1;

train = iddata(data(:,1:2),data(:,3:4),Ts);
% x0 = [-pi/2;0;0;0];

% test  = iddata(Testdaten(:,2),Testdaten(:,1),Ts);
%% Identify state space model from data with n4sid

% Overview over options: https://de.mathworks.com/help/ident/ref/n4sidoptions.html
opt = n4sidOptions('InitialState','estimate','N4Weight','auto','Focus','simulation',...
    'WeightingFilter',[],'EnforceStability',0,...
    'Display','on');
    
[ssm,x0] = n4sid(train,4,opt,'DisturbanceModel','none');


%% Simulate identified model
opt = simOptions('InitialCondition',x0);
[y_train,~,x_train] = sim(ssm,train.u,opt);

% [y_test,~,x_test] = sim(ssm,test.u);

figure;
hold on
plot(data(:,3))
plot(y_train(:,1))
hold off


figure;
hold on
plot(data(:,3))
plot(y_train(:,2))
hold off

figure;
hold on
plot(data(:,3:4))
hold off
% figure;
% hold on
% plot(Testdaten(:,2))
% plot(y_test)
% hold off

%% Save results to python dictionary

Results = struct(...
'A',ssm.A,...
'B',ssm.B,...
'C',ssm.C,...
'D',ssm.D,...
'hidden_train_0',x0,...
'input_train_data',train.u,...
'target_train_data',train.y);

save('RobotManipulator_LSS.mat','Results')
