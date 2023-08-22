clear all

x = zeros( 12, 2 );
file_index = 1;

for control_qubit_index = 1 : 4
  for target_qubit_index = 1 : 4
    if control_qubit_index ~= target_qubit_index
      x( file_index, : ) = [ control_qubit_index target_qubit_index ];
      file_index = file_index + 1;
    end
  end
end

validation_accuracy_all = zeros( 12, 1 );

for index_file = 1 : 12

  eval( [ 'load result_mnist_all_one_cnot' num2str( index_file + 100 ) '.mat net info' ] )

  validation_accuracy_all( index_file ) = info.ValidationAccuracy( end );

end

%%

figure( 1 )

clf

set(gcf,'DefaultAxesFontName','Times');
set(gcf,'DefaultAxesFontSize',10);
set(gcf,'DefaultTextFontName','Times');
set(gcf,'DefaultTextFontSize',10);
set(gcf,'PaperOrientation','portrait')
set(gcf,'PaperType','A4')
set(gcf,'PaperUnits','Inches')
set(gcf,'PaperPosition',[0.25 0.25 8.5/2.54*2 8.5/2.54*1.3])
set(gcf,'Position',[50 50 round(8.5/2.45*96*2) round(8.5/2.54*96*1.3)])
set(gcf,'Color',[1 1 1])

set( gca, 'position', [ 0.1 0.2 0.8 0.7 ] )

bar( validation_accuracy_all )

grid on
box on

axis( [ 0 13 93 94 ] )

set( gca, 'xtick', [] )

for index = 1 : 12

  text( index, 92.9, num2str( x( index, : ) - 1 ), 'Rotation', 90 )

end

ylabel( 'Test set accuracy (%)' )

print -dpng -r300 plot_validation_accuracy_one_cnot.png
