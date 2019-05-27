function [c_str] = clist(list)
%CLIST Summary of this function goes here
%   Detailed explanation goes here
    list = reshape(list.',1,[]);
    c_str = "{ " + sprintf("%0.5ef", list(1)) + sprintf(", %0.5ef", list(2:end)) + " }";
        
    
end

