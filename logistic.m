%% logistic
function log = logistic(var)
    if var<-88
        log = 0;
    elseif var>88
        log = 1;
    else
        log = (1+exp(-1*var))^-1;
    end
end