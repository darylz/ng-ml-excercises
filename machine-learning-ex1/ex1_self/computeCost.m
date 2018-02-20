function J = computeCost(X, y, theta)

    m = size(X, 1);
    errors = computeErrors(X, y, theta);

    % sum of ...:
    J = errors' * errors / (2 * m);

