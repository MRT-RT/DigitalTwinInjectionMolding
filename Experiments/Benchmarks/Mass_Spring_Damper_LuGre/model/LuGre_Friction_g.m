function [g] = LuGre_Friction_g(v)
%STRIBECK_FRICTION Summary of this function goes here
%   Detailed explanation goes here



Fk = 0.2;                    % N
Fs = 0.25;                   % N
vs = 0.002;                  % m/s
sigma_0 = 10E3;


g = 1/sigma_0*( Fk + (Fs-Fk)*exp(-(v/vs)^2) );
g = abs(v)/g;

end

