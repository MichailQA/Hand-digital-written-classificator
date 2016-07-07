function [cost, M] = plotCostWith_M()
  NNStructure = [400 20 35 10];
  numberOfClases = NNStructure(length(NNStructure));
  m = 1000;
  M=[0];
  cost=[0];
  for i = 1:10:m
    [X, y, testSetX, testSetY, Theta] = initializationNN(NNStructure);
    littleSetX = X(1:i,:);
    littleSetY = y(1:i,:);
    
    predict = forwardPropagation(Theta, littleSetX);
    [Theta, c, b]  = backPropagation(littleSetX, littleSetY, Theta, numberOfClases, predict, 30);
    
    predict = forwardPropagation(Theta, littleSetX);
    J = costFunction(predict, littleSetX, littleSetY, numberOfClases);
    
    M = [M i];
    cost = [cost J];
  endfor
endfunction