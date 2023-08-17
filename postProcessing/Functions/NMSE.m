function [nmse] = NMSE(ref,pred)
%Evaulate normalized mean square error
    % mean of reference
    % residual sum of squares
    res     = norm(ref(:)-pred(:),2).^2;
    tot     = norm(ref(:),2).^2;
    nmse = res/tot;
end
