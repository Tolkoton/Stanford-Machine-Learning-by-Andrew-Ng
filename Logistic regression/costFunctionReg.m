function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hoftheta = sigmoid(X*theta);
J += (1 / m) * (-y'*log(hoftheta) - (1 - y)'*log(1-hoftheta)) + (lambda/(2*m))*(theta(2:end)' * theta(2:end));
grad(1) += (1 / m)*(hoftheta - y)' * X(:, 1);
                                                                                
for i = 2: size(theta, 1),
    grad(i) += (1 / m)*(hoftheta - y)' * X(:, i) + (lambda/m)*theta(i);

%grad(2) += (1 / m)*(hoftheta - y)' * X(:, 2) + (lambda/m)*theta(2);
%grad(3) += (1 / m)*(hoftheta - y)' * X(:, 3) + (lambda/m)*theta(3);
%grad(4) += (1 / m)*(hoftheta - y)' * X(:, 4) + (lambda/m)*theta(4);
%grad(5) += (1 / m)*(hoftheta - y)' * X(:, 5) + (lambda/m)*theta(5);





% =============================================================

end
