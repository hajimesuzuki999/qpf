clear all

load result_gtsrb_nn.mat net info

number_of_iteration = length( info.ValidationAccuracy );
number_of_iteration_block = floor( number_of_iteration / 50 );
number_of_iteration = number_of_iteration_block * 50;

validation_accuracy_nn = info.ValidationAccuracy( [ 1 : number_of_iteration_block ] * 50 );
validation_loss_nn = info.ValidationLoss( [ 1 : number_of_iteration_block ] * 50 );
training_loss_nn = mean( reshape( info.TrainingLoss( 1 : number_of_iteration ), 50, number_of_iteration_block ) );

load result_gtsrb_encoding_only.mat net info

number_of_iteration = length( info.ValidationAccuracy );
number_of_iteration_block = floor( number_of_iteration / 50 );
number_of_iteration = number_of_iteration_block * 50;

validation_accuracy_qnn_encoding = info.ValidationAccuracy( [ 1 : number_of_iteration_block ] * 50 );
validation_loss_qnn_encoding = info.ValidationLoss( [ 1 : number_of_iteration_block ] * 50 );
training_loss_qnn_encoding = mean( reshape( info.TrainingLoss( 1 : number_of_iteration ), 50, number_of_iteration_block ) );

load result_gtsrb_one_cnot.mat net info

number_of_iteration = length( info.ValidationAccuracy );
number_of_iteration_block = floor( number_of_iteration / 50 );
number_of_iteration = number_of_iteration_block * 50;

validation_accuracy_qnn_1cnot = info.ValidationAccuracy( [ 1 : number_of_iteration_block ] * 50 );
validation_loss_qnn_1cnot = info.ValidationLoss( [ 1 : number_of_iteration_block ] * 50 );
training_loss_qnn_1cnot = mean( reshape( info.TrainingLoss( 1 : number_of_iteration ), 50, number_of_iteration_block ) );

load result_gtsrb_two_cnot.mat net info

number_of_iteration = length( info.ValidationAccuracy );
number_of_iteration_block = floor( number_of_iteration / 50 );
number_of_iteration = number_of_iteration_block * 50;

validation_accuracy_qnn_2cnot = info.ValidationAccuracy( [ 1 : number_of_iteration_block ] * 50 );
validation_loss_qnn_2cnot = info.ValidationLoss( [ 1 : number_of_iteration_block ] * 50 );
training_loss_qnn_2cnot = mean( reshape( info.TrainingLoss( 1 : number_of_iteration ), 50, number_of_iteration_block ) );

figure( 1 )

clf

set(gcf,'DefaultAxesFontName','Times');
set(gcf,'DefaultAxesFontSize',10);
set(gcf,'DefaultTextFontName','Times');
set(gcf,'DefaultTextFontSize',10);
set(gcf,'PaperOrientation','portrait')
set(gcf,'PaperType','A4')
set(gcf,'PaperUnits','Inches')
set(gcf,'PaperPosition',[0.25 0.25 8.5/2.54*1.3 8.5/2.54*1.3])
set(gcf,'Position',[50 50 round(8.5/2.45*96*1.3) round(8.5/2.54*96*1.3)])
set(gcf,'Color',[1 1 1])

plot(...
  [ 1 : number_of_iteration_block ] * 50, validation_accuracy_nn, '.-', ...
  [ 1 : number_of_iteration_block ] * 50, validation_accuracy_qnn_encoding, '.-', ...  
  [ 1 : number_of_iteration_block ] * 50, validation_accuracy_qnn_1cnot, '.-', ...    
  [ 1 : number_of_iteration_block ] * 50, validation_accuracy_qnn_2cnot, '.-' )

grid on
box on

set( gca, 'xlim', [ 0 8000 ] )
%set( gca, 'ylim', [ 75 100 ] )

xlabel( 'Training iterations' )
ylabel( 'Test set accuracy (%)' )

legend(...
  'NN', ...
  'QPF Encoding only', ...
  'QPF One CNOT', ...
  'QPF Two CNOTs', ...
  'location', 'southeast' )

legend boxoff

print -dpng -r300 plot_result_gtsrb.png
