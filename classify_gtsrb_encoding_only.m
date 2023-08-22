clear all

rng default

load traffic_sign_data_scaled.mat test_labels test_features_scaled train_labels train_features_scaled valid_labels valid_features_scaled

YTrain = categorical( train_labels.' );
XTrain = train_features_scaled;

YTest = categorical( test_labels.' );
XTest = test_features_scaled;

n_train = 34799;
n_test = 12630;

tic

ZTrain = zeros( 16, 16, 4, n_train );

for index_image = 1 : n_train

  current_XTrain = XTrain( :, :, 1, index_image );
  current_output = zeros( 16, 16, 4 );

  for index_j = 1 : 16

    for index_k = 1 : 16

      temp = current_XTrain( 2 * ( index_j - 1 ) + [ 1 2 ], 2 * ( index_k - 1 ) + [ 1 2 ] );
      current_output( index_j, index_k, : ) = circuit_encoding_only( temp( : ) );

    end

  end

  ZTrain( :, :, :, index_image ) = current_output;

end

ZTest = zeros( 16, 16, 4, n_test );

for index_image = 1 : n_test

  current_XTest = XTest( :, :, 1, index_image );
  current_output = zeros( 16, 16, 4 );

  for index_j = 1 : 16

    for index_k = 1 : 16

      temp = current_XTest( 2 * ( index_j - 1 ) + [ 1 2 ], 2 * ( index_k - 1 ) + [ 1 2 ] );
      current_output( index_j, index_k, : ) = circuit_encoding_only( temp( : ) );

    end

  end

  ZTest( :, :, :, index_image ) = current_output;
  
end

toc

layers = [
  imageInputLayer( [ 16 16 4 ] )
  fullyConnectedLayer( 43 )
  softmaxLayer
  classificationLayer];

options = trainingOptions('adam', ...
    'ValidationData', { ZTest, YTest }, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

[ net, info ] = trainNetwork( ZTrain, YTrain, layers, options );

save result_gtsrb_encoding_only.mat net info
