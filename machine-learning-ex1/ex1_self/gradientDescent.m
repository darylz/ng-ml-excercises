function [theta, J_history] = gradientDescent(X, y, theta, alpha, iter_num)

    m = size(X, 1);
    XX = [ones(m,1), X];

    J_history = zeros(m, 1);

    for iter = 1:iter_num
        errors = computeErrors(X, y, theta);
        theta = theta - (alpha/m) * (XX' * errors);

        J_history(iter) = computeCost(X, y, theta);

    end


end

