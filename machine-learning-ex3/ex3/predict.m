function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
X= [ones(m, 1) X];

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

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
first_activator = (Theta1)* X';
second_activator = sigmoid(first_activator);

second_activator = second_activator';
second_activator= [ones(size(first_activator, 2), 1) second_activator];
size(Theta2)
size(second_activator)
third_activator = (Theta2)* second_activator';
third_activator= sigmoid(third_activator);
size(third_activator)
third_activator= third_activator'

% =========================================================================
kind = max(third_activator, [], 2);
size(kind)
for i =1: m

   p(i)= find(third_activator(i,:) == kind(i));
endfor

end
