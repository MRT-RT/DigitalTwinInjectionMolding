function [f_even,f_odd] = calculate_nonlinearity_bins(fs,df,f_exc,mode)

%   fs:     sampling rate   
%   df:     desired frequency resolution
%   f_exc:  excited frequencies
%   mode:   'full' or 'odd' for exciting even and odd (full) or only odd
%           ('odd') frequencies
%%

f = [0:df:fs/2];

if strcmp(mode,'odd')==1

    f_even = f(1:2:end);    % Since only odd bins are excited, all contributions at even frequencies come from the nonlinearity

    f_odd = f(2:2:end);     % odd bins must be reduced by the excited frequencies bins to get the nonlinear contributions
    
    f_odd = f_odd(~ismembertol(f_odd,f_exc));

else
   
    error('This works only, if only odd frequency bins are excited!')
end


end

