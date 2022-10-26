function [F] = LuGre_Friction_F(x)
%STRIBECK_FRICTION Summary of this function goes here
%   Detailed explanation goes here

z  = x(1);
dz = x(2);
m = 0.1;

sigma_0 = 10E3;
sigma_1 = 2*sqrt(sigma_0*m);

F = sigma_0*z + sigma_1*dz;


end

