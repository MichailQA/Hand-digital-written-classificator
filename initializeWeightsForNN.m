function Theta = initializeWeightsForNN(NNStructure)
  for currentLayer = 2:length(NNStructure)
    previusLayer = currentLayer-1;
    Theta{previusLayer} = randn(NNStructure(currentLayer), NNStructure(previusLayer)+1);
  end
end