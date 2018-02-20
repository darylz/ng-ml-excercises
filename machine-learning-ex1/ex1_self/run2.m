
clear; close all;

alpha = 0.03;
iter_num = 200;

D = load('health.txt');
X = D([1:40], [1:4]);
X = featureNormalize(X);
y = D([1:40], 5);

theta = zeros(1+size(X,2), 1);

[theta, J_history] = gradientDescent(X, y, theta, alpha, iter_num);
fprintf('for alpha %f after %d rounds : the theta is below, the cost: %f \n', alpha, iter_num, computeCost(X, y, theta));
pause;
theta

figure;
subplot(2, 2, 1);
plot(X(:,1), computePredictions(X, theta), 'bx');
subplot(2, 2, 2);
plot(X(:,2), computePredictions(X, theta), 'bx');
subplot(2, 2, 3);
plot(X(:,3), computePredictions(X, theta), 'bx');
subplot(2, 2, 4);
plot(X(:,4), computePredictions(X, theta), 'bx');

figure;
plot(J_history, '-');

X_test = D([41:50], [1:4]);
X_test = featureNormalize(X_test);
y_test = D([41:50], 5);

fprintf('test data cost: %f\n', computeCost(X_test, y_test, theta))

