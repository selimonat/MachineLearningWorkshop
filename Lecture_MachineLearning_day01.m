%% Day 1: Linear Algebra Basics and Principal Component Analysis
%
% Concepts covered:
%
% Orthogonality, scalar product, norm, variance, covariance, projection,
% rotation, eigenvector decomposition, principal component analysis.
%
% Usually these are simple concepts (really!), but most of the
% time made difficult with mathematical jargon. I will avoid mathematical
% jargon to make the course as accessible as possible.
% 
% Why linear algebra is important?
%
% Data analysis consists of understanding the data, rather than making
% statistical inferences. 
%
% We view data as vectors occupying a given space (sensor space, pixel
% space, voxel space, etc.). 
% 


subplot(2,1,1);
imshow(imread('~/Desktop/MachineLearningWorkshop//bin/PixelsSpace_BasisVectors.png'));
title('Basis Vectors for Pixel Space')
subplot(2,1,2);
imshow(imread('~/Desktop/MachineLearningWorkshop//bin/PixelsSpace_BasisVectors_02.png'));
title('Weighted basis vectors')
%%
% And as a general rule data occupies only a
% small portion of this space. Linear algebra helps us tremendously in
% order to find how this data is distributed in a space, and if necessary
% to bring the data to a another space which is more efficient, more
% compact. Linear algebra provides us how we can do this type of things.
clf
imshow(imread('~/Desktop/MachineLearningWorkshop//bin/PossibleImages.png'))
title('Natural images occupy only a small portion of the pixel space')
%% Example: Pixel Space of two-pixel images
% An image patch of two pixels can be seen as one vector in a space where
% pixel values are dimensions. 
figure;
v = randn(50,2)*2;

subplot(1,2,1)
imagesc(v);
colorbar;
colormap gray
ylabel('images')
xlabel('pixels')
title('2-pixel images')

subplot(1,2,2);
plot_vectors(v);
xlabel('pixel 1')
ylabel('pixel 2');

% We can think the same way for an usual image of 1600 pixels. This is
% however not possible to plot on a surface. 
%
%
% Q: Can you think of how this would look like when we took instead natural
% images of 2 pixel wide?

%% One fundamental operation: Scalar Product (or Dot Product)
%
% Scalar product takes two vectors and returns one scalar.
%
% Scalar product between two vectors is the sum of all their
% elements after element-wise multiplication. The two vectors MUST have the
% same size.
%
% A*B = SUM ( A(1)*B(1) + A(2)*B(2) + A(3)*B(3) + ... )
%
% In Matlab terms, one can either use matrix multiplication:
% a'*b 
% or element-wise multiplication and summation
% sum(a.*b)
%
%% Meaning of Scalar Product
% 
% We can think of Scalar product as a projection.
clf;
vector         = [1 1];                     % Create a vector
vector         = vector/sqrt(vector*vector') % Make its length equal to one
plot_vectors(vector);                        % plot the vector
hold on 

X              = [0 1];%rand(1,2);                  % Create random points.
plot_vectors(X,'color','r')                  % Plot the random points.
hold on;


scalar_product = vector*X';                  % compute the scalar product.
scaled_vector  = scalar_product*vector;      % scale the vector with the scalar vectors.
try
    plot_vectors(scaled_vector,'color','m')
end
hold on;
plot([X(1) scaled_vector(1)],[X(2) scaled_vector(2)],'r')
axis square
h = legend('Vector V','Data D','ScalarProduct(V,D)*V','OrthogonalProjection');
set(h,'location','northoutside')

% Scalar product between two vectors is proportional to the projection of
% one to the another. 
% It is exactly the perpendicular PROJECTION when the projected vector
% has a unit of 1.
%
% Q: how does this compares to the least-squares distance that we use to
% find the best fitting line in a regression analysis?
%% Scalar product of a vector with itself: a'*a
% Long time ago in the school we learnt how to compute the hypotenus of a
% triangle using Pythagoras formula. For a triangle with edges of 3 and 4
% units, we had
%
% sqrt(3^2 + 4^2) = 5
%
% The square-root of the scalar product of a vector with it self is the
% norm, that is the length of that vector
v = [3 4];
clf;
plot_vectors(v);
v_length = sqrt(v*v');
title(sprintf('The length of vector [%0.1g %0.1g] is %0.3g',v(1),v(2),v_length))

% The scalar product of one vector with itself gives us its length. 
% This makes sense if you think of scalar product as the projection of one
% vector to itself.
%
% Q: Can we now think of a method to transform any vector to a length of 1.
%% Scalar product wrap up.
% The fundamental equation for scalar product between two random vectors:
%
% x dot y = length(x)*length(y)*cos(alpha)
% x dot y / length(x)*length(y) = cos(alpha)
%
% This means three things. The scalar product between x and y is
% proportional to
% (1) the length of the vector x
% (2) the length of the vector y and
% (3) the angle that separates them.
%
% If we have two vectors x and y of length 1, their dot product is equal
% to the cosinus of the angle that separetes them.
%
% For two orthogonal vectors unit vectors, their scalar product will be equal to
% zero.
v = [ 2 4];
v = v./sqrt(v*v');
y = [ -2 5];
y = y./sqrt(y*y')
plot_vectors([v;y])
title(sprintf('The angle between\nthese two vectors is %d degrees',round(rad2deg(acos(v*y')))));
%% To find the angle between any two vectors we need to normalize each
% vector with their length
% 
% v dot y / (norm(v)*norm(y))= cos(alpha)
% v*y' / ( v*v' * y*y' ) = cos(alpha)
% create two unit vectors
v     = rand(1,2);
y     = rand(1,2);
plot_vectors([v;y])
% normalize their length, so that they have unit length
v     = v./sqrt(v*v');
y     = y./sqrt(y*y');
hold on;
plot_vectors([v;y],'color','r')

title(sprintf('The angle between these two vectors is %0.3g degrees',(rad2deg(acos(v*y')))));
%% Linear Transformation
%
% In simple words, a linear transformation is a function that maps one
% position in one space to another one in another space.
%
% Example: One basic transformation is rotation with a given angle.
clf;
v     = [0 1];%randn(3,2);
plot_vectors(v);
% a rotation matrix rotates a vector with a given angle, it is defined as
alpha = deg2rad(33);
R     = [cos(alpha) sin(alpha);-sin(alpha) cos(alpha)]';
v_r   = R*v';
hold on;
plot_vectors(v_r','color','r')
% legend('Vector 1','Vector 2','Transformed 1','Transformed 1');

%% Graphic illustration of transformation in 2D
% we can actually see what a transformation does to the space by looking at
% all the vectors
[y,x]        = meshgrid(-4:.5:4,-4:.5:4);
v            = [x(:) y(:)];

subplot(2,2,1);
plot_vectors(v);
axis square;set(gca,'xlim',[-5 5],'ylim',[-5 5]);
title('Before Transformation');

alpha = deg2rad(12);
R     = [cos(alpha) sin(alpha);-sin(alpha) cos(alpha)]';%rotation
R     = [2 0;0 1]';
v_r   = R*v';%transform

subplot(2,2,2);
plot_vectors(v_r','color','r');
axis square;set(gca,'xlim',[-5 5],'ylim',[-5 5])
title('After Transformation')
%
subplot(2,2,[3 4])
quiver(x(:),y(:),v_r(1,:)'-x(:),v_r(2,:)'-y(:),1,'k')
title('What goes where?')
axis square;set(gca,'xlim',[-5 5],'ylim',[-5 5])
% Q1: Can you come up with another transformation ? 
% Q2: Any idea how we can code your transformation in matrix form ?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%PART 2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Mean of a vector

number_of_trials = 100
x                = randn(number_of_trials,1);
o                = ones(number_of_trials,1);

M                = o'*x/number_of_trials

%% Variance of a vector
% variance(x) = squared sums of x_i around their mean
%             = sum( (x_i - mean(x))^2 )
var(x)                        % variance from Matlab
(x-M)'*(x-M)/number_of_trials % variance
% generally we work we data that is mean corrected
x = x - M;                    % mean centering
x'*x/number_of_trials         

% Variance is the scalar product of a mean-centered vector dividted by the number of points.
%% Covariance between two vectors
% covariance is similar to variance, except two different vectors are multiplied
% instead of one being multiplied with itself.
number_of_trials = 1000;
x                = randn(number_of_trials,1);
x                = x-mean(x);
y                = randn(number_of_trials,1);
y                = y-mean(y);
clf
% plot(x,y,'o');
fprintf('The scalar product between x and y divided by number of points:\n')
x'*y/number_of_trials % 

% a handy way of doing this is to create a data matrix D and then
% multiplying the D with itself. We get both variances and covariances in a
% neat matrix format.

fprintf('Data matrix D multiplied by itself:\n')
D = [x y]; % data matrix
C = D'*D/number_of_trials
C = cov(D)

fprintf('Matlab''s cov() function (uses slightly different normalization):\n')
cov(x,y)

%C = D'*D = [ x*x x*y
%             y*x y*y   ]  / n
%
%         = [ var(x) cov(x,y)
%             cov(y,x) var(y) ]
%
% C is an important matrix! 
% It contains variances on it diagonal entries and covariances in its
% off-diagonal entries.  
%
% Because the co-variance between x and y is the same as y and x, C is a
% symmetric matrix. 
%
% The covariance matrix C summarizes all second-order relationships of the
% present in the data.

%% Few exercices: 
number_of_trials = 5000;
x                = randn(number_of_trials,1);
x                = x-mean(x);
y                = randn(number_of_trials,1);
y                = y-mean(y);
clf;
total_panels     = 4;
panels           = 1:total_panels;%randsample(1:total_panels,total_panels);
h = [];
for i = 1:total_panels
   D = [x y]*randn(2)*1.5;   
   
   h = [h subplot(total_panels,2,i*2-1)];
   plot(D(:,1),D(:,2),'.');
   grid on;box off;xlabel('x');ylabel('y');set(gca,'xlim',[-10 10],'ylim',[-10 10]);
   axis square;
   title(mat2str(i));
   
      
   subplot(total_panels,2,panels(i)*2);   
%    imagesc(cov(D),[-3 3]);
   C = D'*D/number_of_trials
   ImageWithText(C,ceil(C),[-4 4]);
   set(gca,'xtick',[1 2],'ytick',[1 2],'xticklabel',{'x' 'y'},'yticklabel',{'x' 'y'});
   colorbar;
   colormap jet;
%    drawnow;
   axis square;
end


%% Variance across arbitrary directions
%
% When we computed D'*D we are actually computing the variance along the x
% and y-directions.
% Can we compute the variance along arbitrary directions?


% let's create some mean-centered bi-variate data:
number_of_trials = 500;
x                = randn(number_of_trials,1);
x                = x-mean(x);
y                = randn(number_of_trials,1);
y                = y-mean(y);
D                = [x y]*[.05 .2;.2 .3];

% plot the data
subplot(2,1,1);
plot(D(:,1),D(:,2),'.')
set(gca,'xlim',[-10 10],'ylim',[-10 10],'xtick',[-10 0 10],'ytick',[-10 0 10]);
axis equal;axis square;
%
V   = [];
lim = 2;
clf;
for rotation_angle = deg2rad(0:1:180)
    % simply create a unit vector R that is oriented with an angle
    % specified in ROTATION_ANGLE:
    R              = [cos(rotation_angle) sin(rotation_angle)];
    
    %the line above is exactly the same as the commented lines below.
% % %     % create a rotation matrix
% % %     R              = [cos(rotation_angle) sin(rotation_angle);-cos(rotation_angle) sin(rotation_angle)]';
% % %     % rotate the basis vector
% % %     R              = R*[1 0]';
% % %     R              = R./norm(R);
    
    % plot the data cloud and the vector we would like to project the data
    % onto
    subplot(2,1,1)        
    plot(D(:,1),D(:,2),'.')    
    hold on;
    plot_vectors(R,'color','r');
    set(gca,'xlim',[-lim lim],'ylim',[-lim lim],'xtick',[-lim 0 lim],'ytick',[-lim 0 lim]);
    axis square;
    title(sprintf('data and a vector oriented with %d degrees\n',round(rotation_angle/pi*180)));
            
    %project data D to R;   
    projection = R*D';
    
    subplot(2,1,2)    
    V = [V projection*projection'/number_of_trials];%compute the variance       
    plot(V);
    set(gca,'xlim',[0 180]);    
    xlabel('direction of the vector in degrees')
    ylabel('variance')
    drawnow;
end
 % The line above is an important line:
    % projection*projection'/number_of_trials
    %           or equivalently
    %
    %             (R*D)(R*D)'/number_of_trials 
    %
    %           which is equal to
    %
    %              R*D*D'*R'/number of trials
    %
    %           which is equal to
    %               
    %               R * C * R' 
    %
    % where C is the covariance matrix. So the multiplication of a
    % vector with the covariance matrix of the data returns us the variance
    % along that direction.

    
%% We now know how to:
% (1) rotate a vector, 
% (2) project a cloud of dots to on that vector and 
% (3) measure the variance of the projected dots. 
%
% We have all the ingredients to understand PCA:
%% Principle Component Analysis (PCA)
%
% PCA consists of finding directions in the data where variance is
% maximum. The mathematical way of finding the directions of maximum
% variance consists of finding the eigenvectors of the covariance matrix.
%
% Eigenvectors of a matrix M is a vector v, when multiplied by M does
% not change its direction. In math language:
%
% M*v = scalar_lamda*v
%
% In plain language, when matrix M transforms vector V, the result is the
% same vector V scaled by a factor, if V is an eigenvector.
%
% In our situation M is the covariance matrix of data, D. SCALAR_LAMBDA is
% the eigenvalue of the corresponding eigenvector v and equals the variance
% along the corresponding eigenvector.
%
%% Example01: Let's illustrate the effect of eigenvector decomposition step by step.

trow = 3;
tcol = 2;
clf;
% plot the original data and its eigenvectors.
subplot(trow,tcol,1)
plot(D(:,1),D(:,2),'o');
xlabel('x');
ylabel('y');
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

% This shows that PCA is method that rotates and scales the original coordinate
% space into a format which maximizes variance and thereby diagonalizes
% the covariance matrix.
%
% PCA will always work, as long as one can decompose a covariance matrix
% into its eigenvectors. However, the main problem of PCA is whether it
% finds biologically/cognitively meaningful components from the data.
% This is a problem because it will always return some components whether
% they are meaningful or not. 
%
% 
%% Example failure number 1
D = randn(500,2)*[.1 .5;0 .1]';
D = [D ; randn(500,2)*[.1 0;.5 .1]'];
plot_pca_results(D);

%% Example failure number 2:
D = randn(500,2)*[.1 .5;0 .1]';
D = [D ; [5 5]]
plot_pca_results(D)
%% Example failure number 3:
D      = [randn(500,1)*360/pi randn(500,1)*.05+.1];
D      = [D ; randn(500,1)*360/pi randn(500,1)*.05+.6];
[Y,X]  = pol2cart(D(:,1),D(:,2));
D      = [X,Y]
plot(X,Y,'o')
plot_pca_results(D);
%%
D  = D.^2
plot_pca_results(D);
%% Resources:

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% PRACTICAL PART %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%% Sanity checks: Plot your raw image matrix and see if you have everything as expected. 
% Is there an image that is too bright, or too dark? If necessary go back
% to read_preprocess_faces.m and implement the required image processing
% step.

%% Compute the covariance matrix and eigenvector decomposition.

%% Analyze/Visualize your eigenvectors. Plot the first N eigenvectors.

%% Analyze the variance explained by different eigenvectors. 
% Make a cumulative variance analysis that plots the variance explained by n
% first eigenvectors

%% How many eigenvectors you need to explain ~95% of variance in the data?
% Discard the remaining eigenvectors

%% Project your data to the new PCA space and visualize it.

%% Select a random face and project it to the PCA space using only the N eigenvectors and 
%% bring data back to the pixel space. Plot both images side by side.

%% Are there any subspaces that are correlated with different emotional expressions?




