function [u,F,f,f_exc] = RandomPhaseMultisineRandomGrid(fs,df,fband,Nsub,Nrep,mode,energy)
%RANDOMPHASEMULTISINERANDOMGRID Summary of this function goes here
%   fs:     sampling rate   
%   df:     desired frequency resolution
%   Nsub:   number of successive harmonics, of whom the excited ones are
%           randomly picked
%   N:      number of repetitions of the original signal in order to create
%           space between excited frequencies for skirts
%   mode:   'full' or 'odd' for exciting even and odd (full) or only odd
%           ('odd') frequencies
%%

% Error Messages
if (mod(fband(1),df)~=0 | mod(fband(2),df)~=0)
    error('Bandbreite muss Vielfaches der Frequenzauflösung sein!')
elseif (fband(1)<df | fband(1)>fs/2)
    error('Bandbreite außerhalb des darstellbaren Frequenzspektrums!')
elseif Nrep<1
    error('Nrep < 1 !')
end

    
F = zeros(fs/df,1);         % initialize empty spectrum 
F(1)=0;                     % zero mean

if strcmp(mode,'full')==1
    
    for f = 1+fband(1)/df:Nsub:1+fband(2)/df
    
        random = binornd(1,1-1/Nsub,Nsub,1);
        F(f:f+Nsub-1) = random.*exp(-1i*2*pi*rand(Nsub,1));
    
    end
    
    
elseif strcmp(mode,'odd')==1
    
    if(mod(fband(1)/df,2)) == 0            % if start of the frequency band is an even bin
        fstart = 1+fband(1)/df;
        fend   = 1+fband(2)/df;
    else
        fstart = fband(1)/df;
        fend   = fband(2)/df;
    
    end
    for f = 1+fstart : 2*Nsub : 1+fend

        random = binornd(1,1-1/Nsub,Nsub,1);
        F(f:2:f+2*Nsub-1) = random.*exp(-1i*2*pi*rand(Nsub,1));
    
    end
    
end

% Add (Nrep-1) zeros between excited frequencies
Frep = zeros(fs/df * Nrep,1);
Frep(1:Nrep:end) = F;
    
% Normalize signal energy 1
Frep = Frep./sqrt(2*(1/length(Frep)*sum(abs(Frep).^2)));

% Normalize to desired energy
Frep = Frep.*sqrt(energy);
%F = F./sum(abs(F));




% calculate time signal from spectrum
u = ifft(Frep,'symmetric');%*Nrep;
t = 0 : 1/fs : (length(u)-1)*(1/fs);

u = timeseries(u,t);

% calculate single sided amplitude spectrum
F = 2 * abs(Frep(1:fs/df * Nrep/2+1));

% all frequencies up to the Nyquist frequency
f = 0 : df/Nrep : fs/2;

% save excited frequencies
f_exc = f(find(F));



end