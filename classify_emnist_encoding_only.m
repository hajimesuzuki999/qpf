clear all

rng default

load emnist-balanced.mat

number_of_train_images = 112800;
number_of_test_images = 18800;

% 47 classes, 0 to 46
YTrain = categorical( dataset.train.labels );
XTrain = reshape( ( double( dataset.train.images ) ).', 28, 28, 1, number_of_train_images ) / 255;

YTest = categorical( dataset.test.labels );
XTest = reshape( ( double( dataset.test.images ) ).', 28, 28, 1, number_of_test_images ) / 255;

%%
n_train = number_of_train_images    % Size of the train dataset
n_test = number_of_test_images     % Size of the test dataset

tic

ZTrain = zeros( 14, 14, 4, n_train );

for index_image = 1 : n_train
  
  current_XTrain = XTrain( :, :, index_image );
  current_output = zeros( 14, 14, 4 );

  for index_j = 1 : 14

    for index_k = 1 : 14

      temp = current_XTrain( 2 * ( index_j - 1 ) + [ 1 2 ], 2 * ( index_k - 1 ) + [ 1 : 2 ] );
      current_output( index_j, index_k, : ) = circuit_encoding_only( temp( : ) );

    end

  end

  ZTrain( :, :, :, index_image ) = current_output;

end

ZTest = zeros( 14, 14, 4, n_test );

for index_image = 1 : n_test
  
  current_XTest = XTest( :, :, index_image );
  current_output = zeros( 14, 14, 4 );

  for index_j = 1 : 14

    for index_k = 1 : 14

      temp = current_XTest( 2 * ( index_j - 1 ) + [ 1 2 ], 2 * ( index_k - 1 ) + [ 1 : 2 ] );
      current_output( index_j, index_k, : ) = circuit_encoding_only( temp( : ) );

    end

  end

  ZTest( :, :, :, index_image ) = current_output;
  
end

toc

layers = [
  imageInputLayer( [ 14 14 4 ] )
  fullyConnectedLayer( 47 )
  softmaxLayer
  classificationLayer];

options = trainingOptions('adam', ...
    'ValidationData', { ZTest, YTest }, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

[ net, info ] = trainNetwork( ZTrain, YTrain, layers, options );

save result_emnist_encoding_only.mat net info
