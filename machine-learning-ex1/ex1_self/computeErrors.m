function errors = computeErrors(X, y, theta)

    prediction_vector = computePredictions(X, theta);

    errors = prediction_vector - y;

