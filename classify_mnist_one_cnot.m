clear all

rng default

%% load MNIST dataset

fid = fopen( 'train-labels.idx1-ubyte', 'r' );

magic_number = fread( fid, 1, 'int32', 'b' )
number_of_items = fread( fid, 1, 'int32', 'b' )

dat = fread( fid, number_of_items, 'uint8', 'b' );

fclose( fid );

YTrain = categorical( dat );

fid = fopen( 'train-images.idx3-ubyte', 'r' );

magic_number = fread( fid, 1, 'int32', 'b' )
number_of_train_images = fread( fid, 1, 'int32', 'b' )
number_of_rows = fread( fid, 1, 'int32', 'b' )
number_of_columns = fread( fid, 1, 'int32', 'b' )

dat = fread( fid, number_of_rows * number_of_columns * number_of_train_images, 'uint8', 'b' );

fclose( fid );

XTrain = reshape( dat, number_of_rows, number_of_columns, 1, number_of_train_images ) / 255;

% check number of images in each class
number_of_train_images_per_class = zeros( 10, 1 );
for index_digit = 0 : 9
  number_of_train_images_per_class( index_digit + 1 ) = sum( YTrain == categorical( index_digit ) );
end

fid = fopen( 't10k-labels.idx1-ubyte', 'r' );

magic_number = fread( fid, 1, 'int32', 'b' )
number_of_items = fread( fid, 1, 'int32', 'b' )

dat = fread( fid, number_of_items, 'uint8', 'b' );

fclose( fid );

YTest = categorical( dat );

fid = fopen( 't10k-images.idx3-ubyte', 'r' );

disp( 'magic number should be 2051' )

magic_number = fread( fid, 1, 'int32', 'b' )
number_of_test_images = fread( fid, 1, 'int32', 'b' )
number_of_rows = fread( fid, 1, 'int32', 'b' )
number_of_columns = fread( fid, 1, 'int32', 'b' )

dat = fread( fid, number_of_rows * number_of_columns * number_of_test_images, 'uint8', 'b' );

fclose( fid );

XTest = reshape( dat, number_of_rows, number_of_columns, 1, number_of_test_images ) / 255;

%%
n_train = 60000    % Size of the train dataset
n_test = 10000     % Size of the test dataset

tic

ZTrain = zeros( 14, 14, 4, n_train );

for index_image = 1 : n_train
  
  current_XTrain = XTrain( :, :, index_image );
  current_output = zeros( 14, 14, 4 );

  for index_j = 1 : 14

    for index_k = 1 : 14

      temp = current_XTrain( 2 * ( index_j - 1 ) + [ 1 2 ], 2 * ( index_k - 1 ) + [ 1 : 2 ] );
      current_output( index_j, index_k, : ) = circuit_one_cnot( temp( : ) );

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
      current_output( index_j, index_k, : ) = circuit_one_cnot( temp( : ) );

    end

  end

  ZTest( :, :, :, index_image ) = current_output;
  
end

toc

layers = [
  imageInputLayer( [ 14 14 4 ] )
  fullyConnectedLayer( 10 )
  softmaxLayer
  classificationLayer];

options = trainingOptions('adam', ...
    'ValidationData', { ZTest, YTest }, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

[ net, info ] = trainNetwork( ZTrain, YTrain, layers, options );

save result_mnist_one_cnot.mat net info
