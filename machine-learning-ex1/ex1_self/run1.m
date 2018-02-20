
clear; close all;

alpha = 0.3;
iter_num = 100;

D = load('ex1data1.txt');
X = D(:,1);
X = featureNormalize(X);
y = D(:,2);

theta = [0; 0];

J = computeCost(X, y, theta);

fprintf('the cost for theta of [0; 0] is %f <should be 32.07>\n', J);

[theta, J_history] = gradientDescent(X, y, theta, alpha, iter_num);
fprintf('for alpha %f after %d rounds : the theta %f %f, the cost: %f \n', alpha, iter_num, theta(1), theta(2), computeCost(X, y, theta));

figure;
subplot(3, 1, 1);
plot(X, y, 'rx');
hold on;
plot(X, computePredictions(X, theta), 'b-');
hold off;

subplot(3, 1, 2);
plot(J_history, '-');

step_num = 100;
theta0_vals = linspace(0, 2*theta(1), step_num);
theta1_vals = linspace(0, 2*theta(2), step_num);

J_vals = zeros(step_num, step_num);

for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
        theta_tmp = [theta0_vals(i); theta1_vals(j)];
        J_vals(i, j) = computeCost(X, y, theta_tmp);
    end
end

J_vals = J_vals';
figure;
subplot(2, 1, 1);
surf(theta0_vals, theta1_vals, J_vals);

subplot(2, 1, 2);
contour(theta0_vals, theta1_vals, J_vals, logspace(-3, 3, 100));
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
