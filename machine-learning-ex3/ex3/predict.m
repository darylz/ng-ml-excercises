function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1); % mx1, for m samples

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Theta1 : k2 x (n+1)
X = [ones(m, 1), X];   % m x (n+1)

H2 = sigmoid(Theta1 * X')';   % m x k2

% Theta2 : k3 x (k2+1)
H2 =[ones(m, 1), H2];  % m x (k2+1)
H3 = sigmoid(Theta2 * H2')';  % m x k3

[v, p] = max(H3, [], 2);


% =========================================================================


end
