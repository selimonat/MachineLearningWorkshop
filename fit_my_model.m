function [test_error,train_error] = fit_my_model(model,y)

K            = 5;
indices      = crossvalind('Kfold',y,K);
for k = 1:K
    test_indices  =  ismember(indices,k);
    train_indices = ~test_indices;
    
    weights    = model(train_indices,:)\y(train_indices);
    fit        = model*weights;
    
    test_error(k)    = sqrt(mean((fit(test_indices)-y(test_indices)).^2));
    train_error(k)   = sqrt(mean((fit(train_indices)-y(train_indices)).^2));
end
test_error  = mean(test_error);
train_error = mean(train_error);