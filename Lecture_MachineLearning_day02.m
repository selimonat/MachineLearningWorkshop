%% Day 2: Machine Learning Basics and Classification
%
% Our motivation is to learn from data by building a model that generalizes
% best to previously unseen observations.
%
%% Classification 
% let's talk about this equation:
%
% 1*x+2*y = 0   ==> this is an equation of a line
%
% Pair of scalars x and y, that satisfies this equation form a line in 2d.
%
% this is the same as 
% [x y]*[1 2]' = 0;
%
% for any x, there is one y.
%
% For example: 
% if x = 2 then we have -1 for y so that the thing sums to 0.
% we have this equality: x = -2y
% Because y has a coefficient of 2, for each x we have the -.5 as y.
%
% And in general we have y = (-w(1)/w(2))*x
x = -5:.5:5;
w = [1 1];
w = w./sqrt(w*w'); %let's normalize this vectors so that it has unit length
y = x*(-w(1)/w(2));
% this is so important that let's create a little function to get
% corresponding y values for some arbitrary x values.
vector_to_line = @(w,x) x*(-w(1)/w(2));
% Furthermore, let's write another function to plot these x-y pairs.
plot_line      = @(w) plot(linspace(-5,5,100),vector_to_line(w,linspace(-5,5,100)),'k','linewidth',3);
% plot W and its corresponding line.
clf;
plot_vectors(w,'color','r');
set(gca,'xlim',[-5 5],'ylim',[-5 5],'xtick',-5:5,'ytick',-5:5);
hold on;
plot_line(w)
axis square;
h = legend({'Normal vector, w','Points on a line formed by weights in w'});
set(h,'location','northoutside');
% All the points on this black line are orthogonal to the red vector.
% What about other vectors which may lie on both sides of the line?
%
% The interesting observation here is that 
% (1) the vector W is NORMAL to the line that it defines when used as
% coefficients of a line equation. 
% 
% (2) We know that if X is on the line defined by the W, the W*X is 0.

%% Perpendicular distance
% Furthermore, the scalar product of any vector with W gives us a nice
% measure of perpendicular distance to the line. 
% 
X = randn(1,2);
hold on;
plot(X(1),X(2),'bo');
P = w*X'
%
%% Decision boundary
%
% What if we feed W with other values? 
% Points on either side of the decision boundary return values with
% opposite signs. 
%
% We can therefore assign a data point X to a given class depending on
% on which side they are located.
% Let's illustrate this further:
data           = linspace(-5,5,100)';
w              = randn(1,2);
w              = w/sqrt(w*w');
clf;
plot_vectors(w,'color','r');hold on;
plot_line(w);
axis equal;set(gca,'xlim',[-1 1],'ylim',[-1 1]);
xlabel('X(1)')
ylabel('X(2)')
hold on
pause(1)
%
[y,x]          = meshgrid(-1:.1:1,-1:.1:1)
data           = [x(:) y(:)];%each row is one imaginary data point.
for i = 1:size(data,1)
    X = data(i,:);
    %the sign of the projection determines the class of a datapoint.    
    %the value of w*X gives a signed measure of the distance from the
    %decision boundary.
    if w*X' > 0 %!!!!! The most important line
        plot(X(1),X(2),'r+');
    else
        plot(X(1),X(2),'b+');
    end
end
hold off;
h= legend('Normal vector','Decision Boundary')
set(h,'location','bestoutside')

%% Finding the best decision boundary to classify observations
% So now we have a way of conceptualizing a decision boundary. 
% 
% We have to come up with a way of finding the best decision boundary to
% separate classes.
% 
C1 = randn(10,2)-2;% class 1
C2 = randn(10,2)+2;% class 2

D = [C1;C2];
T = [ones(length(C1),1);-ones(length(C1),1)];
%% Perceptron Algorithm
% Let's look at the Perceptron algorithm, which was developped at the 60s. 
% Interesting read: https://en.wikipedia.org/wiki/Perceptron
%

lim  = round(max(abs(D(:)))*1.5);
w    = [1 0.2];
w    = w./sqrt(w*w');
rate = .1;
while 1
    for i = 1:size(D, 1)
        %compute class and predictive value (unused here)
        %     [c, p] = perceive(x(i,:), w);                
        %
        true_class      = T(i);            % true class for point i.
        predicted_class = sign(w*D(i,:)'); % predicted class for point i.                
        
        w_old = w;
        %if there is a mismatch        
        if predicted_class ~= true_class            
            w     = w + (rate * D(i,:) * true_class);           
        end        
        
        clf;
        plot_vectors(w_old);
        hold on;        
        set(gca,'xlim',[-lim lim],'ylim',[-lim lim],'xtick',[-lim 0 lim],'ytick',[-lim 0 lim]);
                        
        plot_vectors(w,'color','g');
        set(gca,'xlim',[-lim lim],'ylim',[-lim lim],'xtick',[-lim 0 lim],'ytick',[-lim 0 lim]);
        hold on;        
                
        plot_vectors(D(i,:),'color','b');
        set(gca,'xlim',[-lim lim],'ylim',[-lim lim],'xtick',[-lim 0 lim],'ytick',[-lim 0 lim]);
        hold on;
        
        plot(D(T == 1,1),D(T == 1,2),'bp','markersize',20);        
        hold on;
        
        plot(D(T == -1,1),D(T == -1,2),'rs','markersize',20);        
        set(gca,'xlim',[-lim lim],'ylim',[-lim lim],'xtick',[-lim 0 lim],'ytick',[-lim 0 lim]);
        
        plot_line(w_old);
        hold on        
        axis square;hold on;
        legend({'Normal Vector','New Normal Vector','Current Data Point'},'location','northoutside');
        drawnow;
        pause
    end
end
% Drawbacks:
% The solutions are not unique.
% Data must be perfectly separable.
% Does not scale really good to millions of data points.
%% 
%% Second Approach: Least-squares.
%
% Regression = Sum of all distances between a data point and the line.
% we find the line that minimizes this error term. The solution is
% analytically computed and unique.
%
% Now we have a slightly different situation where the error is not anymore
% a goodness of fit, but goodness of classification.
% 
% We aim to find the line classifies these two classes
%%
C1 = randn(10,2)*.2-1;% class 1
C2 = [randn(10,1)*.2+3 randn(10,1)*.2+1];% class 2

D = [C1;C2];
T = [ones(length(C1),1);-ones(length(C1),1)];
%%
w = [1 0];
clf;
error = [];
for angle = 1:360
  
        alpha     = deg2rad(angle);
        R         = [cos(alpha) sin(alpha);-sin(alpha) cos(alpha)]';
        w_rotated = R*w';
        W         = [w_rotated w_rotated ];
        %
        subplot(1,2,1);
        plot(C1(:,1),C1(:,2),'ro')        
        hold on;
        plot(C2(:,1),C2(:,2),'bo')
        plot_vectors(w_rotated','color','r');
        hold on
        plot(-5:0.1:5,vector_to_line(w_rotated,-5:0.1:5),'k','linewidth',4)        
        hold off
        axis equal;set(gca,'xlim',[-5 5],'ylim',[-5 5])            
        
%         (D*w_rotated-T)'*(D*w_rotated-T)        
        error(angle) = (D*w_rotated-T)'*(D*w_rotated-T);
%         error(angle) = trace((W*D' - T)'*(W*D' - T))
        
        subplot(1,2,2);        
        plot(error,'o-');
        xlabel('orientation');
        ylabel('error function');
        axis square;        
        drawnow;
        pause(0.1);
end
%%
[~,i]     = min(error);
alpha     = deg2rad(i-1);
R         = [cos(alpha) sin(alpha);-sin(alpha) cos(alpha)]';
w         = [1 0];
w_rotated = R*w';

subplot(1,2,1);
plot(C1(:,1),C1(:,2),'ro')        
hold on;
plot(C2(:,1),C2(:,2),'bo')
plot_vectors(w_rotated','color','r');
hold on
plot(-5:0.1:5,vector_to_line(w_rotated,-5:0.1:5),'k','linewidth',4)
hold off
axis equal;set(gca,'xlim',[-5 5],'ylim',[-5 5])            
%% how to solve this analytically
w_analytical = D\T; %same as w = lsqlin(D, T)
% w_analytical/sqrt(w_analytical'*w_analytical);
hold on;
plot_vectors(w_analytical','color','m');
set(gca,'xlim',[-5 5],'ylim',[-5 5])            
hold on;
plot(-5:0.1:5,vector_to_line(w_analytical,-5:0.1:5),'g--','linewidth',4)
hold off;
%true class, predicted class
%classification error term
% error = true_class - predicted_class
%% one drawback of the least-square method
C1 = randn(50,2)+repmat([2 0],50,1);
C2 = randn(50,2)+repmat([-2 0],50,1);;
C1 = [C1;randn(50,2)*.1+repmat([15 5],50,1)];
clf;
plot(C1(:,1),C1(:,2),'ro')
hold on;
plot(C2(:,1),C2(:,2),'bo')
hold off;
axis equal
set(gca,'xlim',[-10 10],'ylim',[-10 10])
grid on;
% let create another matrix which identifies the class
X = [C1;C2];
T = [[ones(length(C1),1); -ones(length(C2),1)]];
%
z = lsqlin([X ones(length(X),1)], T);
w_analytical = T\[X ones(length(X),1)];
hold on;
plot_vectors(w_analytical,'color','r');
set(gca,'xlim',[-20 20],'ylim',[-20 20]);
hold on;
plot(-5:0.1:5,vector_to_line(w_analytical,-5:0.1:5),'k','linewidth',4)
hold off;
%too correct data points influences the decision boundary.
%% Support Vector Machines:
%distance of a point to the line
% vector_to_line = @(w,x) x*(-w(1)/w(2));
C1 = randn(50,2)+repmat([5 5],50,1);
C2 = randn(50,2)+repmat([-5 -5],50,1);
T  = [[ones(length(C1),1); -ones(length(C2),1)]]
X  = [C1;C2];
%%
clf;
plot(C1(:,1),C1(:,2),'ro')
hold on;
plot(C2(:,1),C2(:,2),'bo')
axis equal
set(gca,'xlim',[-10 10],'ylim',[-10 10])
grid on;
w = [0 1];
w = w/sqrt(w*w');
hold on;
plot_line([w 1]);
hold on;
plot_vectors(w,'color','r');
hold on;
%
distances = w*X'.*T'
[min_distance,i] = min(distances);
plot(X(i,1),X(i,2),'+m','markersize',20)
%
hold on;
plot_normal_and_line([ w min_distance])
plot_normal_and_line([ w -min_distance])
%% How do we select among many models? Model Selection
%
% In machine learning we are interested in making good generalizations. 
%
% This means not only,
%
% (1) building a model that learns from data but also 
% (2) estimating as good as possible how it can generalize to other
%     datasets that we have not so far observed.
%
% Note that:
% This is in contradiction to classical analysis where you fit a
% model to all the data you have. And make model selection using things
% like AIC, BIC, etc. (based on assumptions).
% 
% (1) and (2) impose significant changes in the analysis pipeline for we need
% a way to validate our models.
%
% 
% CROSS-VALIDATION:
%
% We generally split our data into training and testing subsets. We
% fine-tune our model using the training set and measure its 
% performance in the test set. This is called cross-validation.
%
% Our aim is to learn those features of the data set that generalizes to
% previously unseen data. We should not learn features that are specific to
% the currently held data set.
% The purpose of cross-validation is model checking ==> We want to se
%% Different types of cross-validation
%
% k-fold CV:
% Partitions data into k randomly chosen subsets (or folds) of
% roughly equal size. One subset is used to validate the model trained
% using the remaining subsets. This process is repeated k times such that
% each subset is used exactly once for validation.   
%
% Holdout:
% Partitions data into exactly two subsets (or folds) of specified
% ratio for training and validation. 
%
% Leaveout
% Partitions data using the k-fold approach where k is equal to
% the total number of observations in the data. Also known as leave-one-out
% cross-validation.  
%
% Repeated random sub-sampling 
% Performs Monte Carlo repetitions of randomly partitioning data and
% aggregating results over all the runs.
%
% Stratify: 
% Partitions data such that both training and test sets have
% roughly the same class proportions in the response or target.
%
%
% Here is an example of 5-fold cross-validation.
%
% DATASET   |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%| 
%
% batch1 =  |  Train       | Train      |  Train    |           Train |           Test|
% batch2 =  |  Train       | Train      |  Train    |           Test  |           Test|
% batch3 =  |  Train       | Train      |  Test     |           Train |           Test|
% batch4 =  |  Train       | Test       |  Train    |           Train |           Test|
% batch5 =  |  Test        | Train      |  Train    |           Train |           Test|
%
% For each of the folds one can compute a generalization metric, and
% average them.
%
%
%% The practical session:
%
% For this task, we have two datasets available. You may want to continue
% working with the previous dataset of (1) emotional faces, or alternatively
% download the (2) eye-movement dataset. Eye-movement dataset contains
% eye-movement recordings of humans while they were looking at 8 different
% faces. For dataset (1), an interesting task would be training and testing
% an SVM classifier that predicts the emotional expression of the face. For
% dataset (2), an interesting task would be predicting the participant's
% identity based on eye-movement recordings. You may also want to use an
% SVM for any arbitrary task of your imagination. 
%
% Your task is too choose the optimal number of principal components that are
% required to reach the best classification performance. I would like to
% stir up some discussion on how classification performance depends on the
% number of principal components that are used during learning.

clear all;
path_to_data = '~/Desktop/MachineLearningWorkshop/';
load(fullfile(path_to_data,'face_database_ready.mat'));
%% Sanity checks: Plot your raw image matrix and see if you have everything as expected. 
% Is there an image that is too bright, or too dark? If necessary go back
% to read_preprocess_faces.m and implement the required image processing
% step.
clf;
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
%Q: on which dataset we compute the PCA analysis?
%% Project faces to the eigenspace and whiten
P     = e(:,1:250)'*im';
v2    = diag(v(1:250));
v_inv = sqrt(inv(v2));
Pw    = P'*v_inv;
%%