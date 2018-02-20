function X_norm = featureNormalize(X)

    X_norm = (X - mean(X, 1)) ./ std(X, 1);

