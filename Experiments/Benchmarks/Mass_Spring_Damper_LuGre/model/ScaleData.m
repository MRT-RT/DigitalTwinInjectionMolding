function [data,scaler] = ScaleData(data, scaler)
%SCALEDATA scales data either to the range of [-1,1] or [a,b]

if isempty(scaler)
    a = min(data);
    b = max(data);
    
    % Scale between -1,1
    data = 2 * (data-a) / (b-a) -1;
    
    % Scale between 0,1
    %data = (data-a) / (b-a);
    
    scaler = [a,b];

else
    a = scaler(1);
    b = scaler(2);
    
    % Scale between -1,1    
    data = 2 * (data-a) / (b-a) -1;
    
    % Scale between 0,1
    %data = (data-a) / (b-a);

end


end

