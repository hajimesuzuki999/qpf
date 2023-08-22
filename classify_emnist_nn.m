clear all

rng default

%% load EMNIST dataset

load emnist-balanced.mat

number_of_train_images = 112800;
number_of_test_images = 18800;

% 47 classes, 0 to 46
YTrain = categorical( dataset.train.labels );
XTrain = reshape( ( double( dataset.train.images ) ).', 28, 28, 1, number_of_train_images ) / 255;

% check number of images in each class
number_of_train_images_per_class = zeros( 47, 1 );
for index_class = 1 : 47
  number_of_train_images_per_class( index_class ) = sum( YTrain == categorical( index_class - 1 ) );
end

YTest = categorical( dataset.test.labels );
XTest = reshape( ( double( dataset.test.images ) ).', 28, 28, 1, number_of_test_images ) / 255;

%%

layers = [
  imageInputLayer( [ 28 28 1 ] )
  fullyConnectedLayer( 47 )
  softmaxLayer
  classificationLayer];

options = trainingOptions('adam', ...
    'ValidationData', { XTest, YTest }, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

[ net, info ] = trainNetwork( XTrain, YTrain, layers, options );

save result_emnist_nn.mat net info
