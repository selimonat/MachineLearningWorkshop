%% Practical Session: EigenFaces, PCA analysis of faces with emotional expressions
%
% We will use a dataset of faces pictures with emotional expressions. 
% We will compute PCA on this data set. The primarily task is to get
% hands-on experience on conducting PCA and understand how what we have
% learnt in two dimensions generalizes to N dimensions.
%
% Data:
% You are receiving a 3D matrix organized as (pixel_y,pixel_x,image),
% and a binary matrix of emotional ratings (image,emotional_dimension).

%% Preprocessing.
% Before you actually conduct a PCA, please think of the following questions:
% (1) How many dimensions do you have? 
% (2) How big a size would the resulting covariance matrix occupy in the
% memory of your computer?
% (3) What kind of preprocessing steps would you envisage to make this problem
% more tractable ? 
% (4) What kind of other preprocessing steps you can think of to make PCA
% more sensitive with respect to finding interesting dimensions in the
% data. 
% 
%% Read the face and emotion data. 
% To speed up things I have written read_preprocess_faces.m function which
% does everything except preprocessing. Use that file to import data to
% Matlab and code your preprocessing directly in it.
%
clear all;
path_to_data = '~/Desktop/MachineLearning/';
load(fullfile(path_to_data,'face_database_ready.mat'))
%% Sanity checks: Plot your raw image matrix and see if you have everything as expected. 
% Is there an image that is too bright, or too dark? If necessary go back
% to read_preprocess_faces.m and implement the required image processing
% step.
imagesc(im);
colorbar;
ylabel('images');
xlabel('dimensions');
%% Compute the covariance matrix and eigenvector decomposition.
C     = cov(im);            % compute covariance matrix
[e v] = eig(C);             % eigenvector decomposition
%flip everything left to right so that it is more intuitive to index.
v     = flipud(diag(abs(v)));   
e     = fliplr(e);
e_inv = inv(e);
%% Analyze/Visualize your eigenvectors. Plot the first N eigenvectors.
clf;
total_vector = 25;                                                   %number of eigenvectors to plot
for i = 1:total_vector
    h = subplot(ceil(sqrt(total_vector)),ceil(sqrt(total_vector)),i); 
    imagesc(reshape(e(:,i),image_height,image_width))
    title(sprintf('Vec: %d\nValue: %d',i,round(sqrt(v(i)))),'fontsize',14);
    colormap gray;
    axis off;
    axis image;
    subplotChangeSize(h,.01,.01);
    drawnow;
end
%% Analyze the variance explained by different eigenvectors. 
% Make a cumulative variance analysis that plots the variance explained by n
% first eigenvectors
clf
plot(100*cumsum(v)/sum(v),'o-');
ylabel('Explained Variance (%)');
xlabel('Eigenvectors');
axis tight;
box off;
grid on;
%% How many eigenvectors you need to explain ~95% of variance in the data?
% Discard the remaining eigenvectors
total_variance          = 95;                                           % how much variance to be explained
first_above_95          = find(100*cumsum(v)/sum(v) > total_variance,1);% find the first eigenvector that is explains more than total_variance
fprintf('The first %d eigenvectors explains %d of variance...\n',first_above_95,total_variance);
% How much compression did you achieve? Let's define compression as the
% ratio of dimensions before and after.
compression             = (1-first_above_95/size(im,2))*100
%% Project your data to the new PCA space and visualize it.
clf;
P  = e(:,1:100)'*im';
imagesc(P);colormap gray;
ylabel('# images');
xlabel('# PCA dimensions');
title('Data in the PCA space');
%% Select a random face and project it to the PCA space using only the N eigenvectors and 
%% bring data back to the pixel space. Plot both images side by side.
image_index      = 1;
for t                = 1:10:400;
    v2               = v;
    v2(t+1:end)      = 0;
    face_pca_space   = e'*im(image_index,:)';
    face_pca_space(t+1:end) = 0;
    %
    face_pixel_space = e_inv'*face_pca_space;
    face_pixel_space = reshape(face_pixel_space,image_height,image_width);
    % imagesc(reshape(face_pixel_space,image_height,image_width));
    %
    clf;
    subplot(1,2,1);
    original_face    = reshape(im(image_index,:),image_height,image_width);
    imagesc(original_face);
    axis image;
    title('original face');
    
    subplot(1,2,2)
    imagesc(face_pixel_space);
    title(sprintf('Reconstruction with %d eigenvectors',t))
    axis image;
    drawnow;
    pause(0.1);
end
%% (4) Are there any subspaces that are correlated with different emotional expressions?
% Emotional signature
v2     = diag(v(1:100));
v_inv = inv(sqrt(v2));
e     = e(:,1:100);
P     = e'*im';
Pw    = v_inv*P;

%%
Pemo    = Pw*emotional_code;
imagesc(Pemo)
P3      = e_inv(1:100,:)'*Pemo;
%% Identify an emotional subspace and inject it to a neutral face, does this change the emotional expression of a face?
P4 = im'*emotional_code;
n = 2
subplot(1,2,1);
imagesc(reshape(P3(:,n),image_height,image_width));axis image;
subplot(1,2,2);
imagesc(reshape(P4(:,n),image_height,image_width));axis image;

