function plot_pca_results(D);

trow = 3;
tcol = 2;
clf;
number_of_trials = length(D);
lim              = ceil(max(abs(D(:)))*1.3);
% plot the original data and its eigenvectors.
subplot(trow,tcol,1)
plot(D(:,1),D(:,2),'o');
xlabel('x');
ylabel('y');
set(gca,'xlim',[-lim lim],'ylim',[-lim lim],'xtick',[-lim 0 lim],'ytick',[-lim 0 lim]);
title('Data and eigenvectors of D''*D');

% Visualize its covariance matrix
subplot(trow,tcol,2);
C     = D'*D/number_of_trials;
imagesc(C,[0 .2]);colorbar;axis square
set(gca,'xtick',[1 2],'xticklabel',{'x' 'y'},'ytick',[1 2],'yticklabel',{'x' 'y'})
title('covariance matrix C')

% Compute the covariance matrix of the newly projected data.
[e v] = eig(C);
subplot(trow,tcol,1);
hold on;
plot_vectors(e','color','m')
set(gca,'xlim',[-lim lim],'ylim',[-lim lim],'xtick',[-lim 0 lim],'ytick',[-lim 0 lim]);
title('Eigenvectors of C');

% Project data to the new eigenvector space
subplot(trow,tcol,3);
D2 = e'*D';
plot(D2(1,:),D2(2,:),'o');
set(gca,'xlim',[-lim lim],'ylim',[-lim lim],'xtick',[-lim 0 lim],'ytick',[-lim 0 lim]);
axis square;
xlabel('eigenvector 1');
ylabel('eigenvector 2');
box off;grid on;
axis square;
title('Data in EigenSpace')

% Visualize the new covariance matrix (which are the eigenvalues)
subplot(trow,tcol,4);
imagesc(v,[0 .2]);colorbar;
set(gca,'xtick',[1 2],'xticklabel',{'eig1' 'eig2'},'ytick',[1 2],'yticklabel',{'eig1' 'eig2'})
title(sprintf('Covariance\nin\nEigenspace (Eigenvalues)')); 
axis square;

% Whiten: Normalize projected data with their eigenvalues.
% 
% Whiten: Make data white, that is make contributions of dimensions equal.
% In analogy to frequencies, when all frequencies in the light contribute
% equally the resulting color is white.
subplot(trow,tcol,5);
D3 = v^(-1/2)*e'*D';
plot(D3(1,:),D3(2,:),'o');
set(gca,'xlim',[-lim lim],'ylim',[-lim lim],'xtick',[-lim 0 lim],'ytick',[-lim 0 lim]);
axis square;
xlabel('eigenvector 1');
ylabel('eigenvector 2');
box off;grid on;
axis square;
title('Data in EigenSpace')

% Plot the covariance matrix of the whitened data
subplot(trow,tcol,6);
imagesc(D3*D3'/number_of_trials,[0 .2]);colorbar;
set(gca,'xtick',[1 2],'xticklabel',{'eig1' 'eig2'},'ytick',[1 2],'yticklabel',{'eig1' 'eig2'})
title(sprintf('Covariance\nin\nEigenspace (Eigenvalues)')); 
axis square;