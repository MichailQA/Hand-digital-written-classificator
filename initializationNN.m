function [X, y, testSetX, testSetY, Theta] = initializationNN(NNStructure)
  load('ex4data1.mat')
  
  random = randperm(size(X,1));
  random = random(1:2000);
  
  testSetX = X(random,:);
  testSetY = y(random,:);
  
  X(random,:) = [];
  y(random,:) = [];
  
  random = randperm(size(X,1));
  X = X(random,:);
  y = y(random,:);
  
  Theta = initializeWeightsForNN(NNStructure);
endfunction