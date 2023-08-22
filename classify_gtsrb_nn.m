clear all

rng default

load traffic_sign_data_scaled.mat test_labels test_features_scaled train_labels train_features_scaled valid_labels valid_features_scaled

YTrain = categorical( train_labels.' );
XTrain = train_features_scaled;

YTest = categorical( test_labels.' );
XTest = test_features_scaled;

tic

layers = [
  imageInputLayer( [ 32 32 1 ] )
  fullyConnectedLayer( 43 )
  softmaxLayer
  classificationLayer];

options = trainingOptions('adam', ...
    'ValidationData', { XTest, YTest }, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

[ net, info ] = trainNetwork( XTrain, YTrain, layers, options );

toc

save result_gtsrb_nn.mat net info
