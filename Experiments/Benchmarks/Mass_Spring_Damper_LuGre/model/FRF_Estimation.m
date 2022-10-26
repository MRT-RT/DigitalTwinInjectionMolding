clear all
close all
clc
%% Load experimental data
Exp1    = load('TimeVaryingFeedforward_Exp1');
Exp2    = load('TimeVaryingFeedforward_Exp2');
%% specify hyperparameters for estimation

Np = 2;     % Order of series expansion / number of 'branches'
ke = 1;     % Number of neighbouring excited frequencies to consider for locally modelling the output DFT spectrum
Na = 1;     % Order of polynomial to estimate contribution of skirts from excited frequencies outside of the local frequency band
N = length(Exp1.U2);

%%

%[G,B] = TV_FRF_Estimation_v2(U2(1:end/2+1),Y2(1:end/2+1),T2,fs,Np,f_exc_bin);

% Use Data from second experiment to estimate the model
[G2,B] = TV_Sliding_FRF_Estimation(Exp2.U2,Exp2.Y2,Exp2.T2,Exp2.fs,Np,...
    Na,ke,Exp2.f_exc_bin);

% Check if model describes measured ouput satisfactory
Y2 = TV_Sliding_FRF_Evaluation(G2,Exp2.U2(1:end/2+1));

%%
% Interpolate estimated model to get a model for the first experiment
G1 = SlidingFRF_Interpolation(G2,Exp1.f_exc_bin);
%%
% Check if model describes measured ouput satisfactory
Y1 = TV_Sliding_FRF_Evaluation(G1,Exp1.U2(1:end/2+1));

%% Plot estimated output spectrum of Experiment 2 using estimated G2
figure
hold
for i = 1:length(G2)
    plot(G2{i}.band,db(abs(Exp2.Y2(G2{i}.band))),'bx')
    plot(G2{i}.band,db(abs(Y2(G2{i}.band))),'x')
end


figure
hold
for i = 1:length(G2)
    plot(G2{i}.band,angle(Exp2.Y2(G2{i}.band)),'bx')
    plot(G2{i}.band,angle(Y2(G2{i}.band)),'x')
end


%% Plot estimated output spectrum of Experiment 1 using interpolated G1
figure
hold
for i = 1:length(G1)
    plot(G1{i}.band,db(abs(Exp1.Y2(G1{i}.band))),'bx')
    plot(G1{i}.band,db(abs(Y1(G1{i}.band))),'x')
end


figure
hold
for i = 1:length(G1)
    plot(G1{i}.band,angle(Exp1.Y2(G1{i}.band)),'bx')
    plot(G1{i}.band,angle(Y1(G1{i}.band)),'x')
end

ym = ifft(Exp1.Y2,'symmetric');                                                 % noisy measured output 
ykc = ifft([Y1,zeros(1,length(Exp1.Y1)-length(Y1))],'symmetric');                % estimated output

%%
figure;
hold on
plot(ym)
plot(ykc)
hold off

% figure
% hold on
% plot(f2,db(abs(Y2(1:end/2+1))),'x')
% plot(f2,db(abs(Y)),'x')

%plot(2*pi*f2(f_exc_bin),db(abs(params(1:end/2))),'x')