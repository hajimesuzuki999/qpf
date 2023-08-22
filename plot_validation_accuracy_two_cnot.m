clear all

validation_accuracy_all = zeros( 24, 1 );

for index_file = 1 : 24

  eval( [ 'load result_mnist_all_two_cnot' num2str( index_file + 100 ) '.mat net info' ] )

  validation_accuracy_all( index_file ) = info.ValidationAccuracy( end );

end

x = flipud( perms( [ 1 2 3 4 ] ) );

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

bar( flipud( validation_accuracy_all ) )

grid on
box on

axis( [ 0 25 92.5 95.5 ] )

set( gca, 'xtick', [] )

for index = 1 : 24

  text( index, 91.9, num2str( x( index, : ) - 1 ), 'Rotation', 90 )

end

ylabel( 'Test set accuracy (%)' )

print -dpng -r300 plot_validation_accuracy_two_cnot.png
