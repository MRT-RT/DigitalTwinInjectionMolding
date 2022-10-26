close all
clear
clc


%% System parameters of the mass spring damper system
m = 0.1;                    % kg      
d = 4;                      % Ns/m
k_0 = 20;                  % N/m   stiffness at beginning of experiment
xInitial = [0.0,0];

%% Sampling rate, bandwidth, 'space' between excited frequencies, 
% number of repetitions for skirts

fs = 200;           % Sampling rate in Hz
fband = [0.1,20];   % Bandwidth in Hz
df = 0.1;           % frequency resolution in Hz
Nsub = 5;           % consecutive harmonics to be chosen from which to excite
Nrep = 4;           % Number of repetitions, equals bins between excited frequencies  
mode = 'odd';       % excite only odd or all frequencies  
energy = 1;         % desired signal energy 



%%

T_s = 1/fs;        % Sampling time  
N = fs/df * Nrep;    % Number of samples
T_sim = N*T_s-T_s;  % Simulation time

delta = d/(2*m);
w0 = sqrt(k_0/m);
D_lehr = d/(2*m*w0);

% State space matrices
A       = [ 0       1
           -k_0/m    -d/m];

B       = [ 0
           k_0/m];

C       = [1  0];

D       = [0];

A_nl    = [0    
           -1/m ];
% A_nl    = [0    
%            0 ];       
       
[b,a] = ss2tf(A,B,C,D);

G = tf(b,a);
bode(G);
