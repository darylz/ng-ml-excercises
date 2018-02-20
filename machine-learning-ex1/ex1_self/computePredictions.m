function predictions = computePrediction(X, theta)

    % sample number is the row number:
    m = size(X, 1);

    % appending x0 to the left:
    XX = [ones(m, 1), X];

    predictions = XX * theta;

