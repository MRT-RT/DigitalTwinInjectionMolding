function [u,y,T] = GetRidofTransient(u,y,Ndel,fs,df)
%GETRIDOFTRANSIENT Summary of this function goes here
%   u:          Input time series   
%   y:          Output time series  
%   T:          length of truncated time series in time
%   Ndel:       number of periods to delete
%   Nperiod:    number of observations per period

Nperiod = fs/df;

N_throwaway = Ndel * Nperiod;                                              % Throw away samples from first period

T = (length(u.data)-N_throwaway) * 1/fs - 1/fs;                            % -1/fs since we start at 0

u = timeseries(u.Data(N_throwaway+1:end), u.Time(N_throwaway+1:end));
y = timeseries(y.Data(N_throwaway+1:end), y.Time(N_throwaway+1:end));

end

