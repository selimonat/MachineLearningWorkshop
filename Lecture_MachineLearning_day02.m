%% Day 2: Machine Learning Basics:
%
% Learn from data and generalize to unseen observations.
%
% Supervised
% k-Nearest Neighbors
% Linear Regression
% Logistic Regression
% Support Vector Machines (SVMs)
% Decision Trees and Random Forests
% Neural networks
%
%
% Unsupervised
% ? Clustering
%   ? k-Means
%   ? Hierarchical Cluster Analysis (HCA) ? Expectation Maximization
% ? Visualization and dimensionality reduction ? Principal Component Analysis (PCA) ? Kernel PCA
%   ? Locally-Linear Embedding (LLE)
%   ? t-distributed Stochastic Neighbor Embedding (t-SNE)
% ? Association rule learning
%   ? Apriori ? Eclat
%
%Model: A specific mathematic expression.
%For example, a linear model, or a quadratic model
%
%A model instance: One model fitted to a specific set of datasets.
%
%
%% Model Selection:
%
% In machine learning we are interested in making good generalizations. This
% means building a model that learns from data and estimating as good as possible how it
% generalizes to other datasets that we have not so far observed.
%
% This is in stark contradiction to classical analysis where you fit a
% model to all the data you have. And make model selection based on strong
% assumptions.
% 
% For this reasons, we have different methodologies to cope with this
% demand: Cross-validation.
%
% We generally split our data into training and testing
% subsets. We fine-tune our model using the training set and measure its
% performance in the test set. This is called cross-validation.
%
% Our aim is to learn those features of the data set that generalizes to
% previously unseen data. We should not learn feature that are specific to
% the currently held data set.
% the purpose of cross-validation is model checking ==> We want to se
%% Different types of cross-validation
%
% k-fold: Partitions data into k randomly chosen subsets (or folds) of
% roughly equal size. One subset is used to validate the model trained
% using the remaining subsets. This process is repeated k times such that
% each subset is used exactly once for validation.   
%
% Holdout: Partitions data into exactly two subsets (or folds) of specified
% ratio for training and validation. 
%
% Leaveout: Partitions data using the k-fold approach where k is equal to
% the total number of observations in the data. Also known as leave-one-out
% cross-validation.  
%
% Repeated random sub-sampling: Performs Monte Carlo repetitions of
% randomly partitioning data and aggregating results over all the runs.  
%
% Stratify: Partitions data such that both training and test sets have
% roughly the same class proportions in the response or target.
%
% Resubstitution: Does not partition the data; uses the training data for
% validation. Often produces overly optimistic estimates for performance
% and must be avoided if there is sufficient data.  
%
% Here is an example of 5-fold cross-validation.
%
%           |                     COMPLETE DATA SET                                   | 
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
%% Example: 
% Generate some data with noise:
number_of_points = 70;
noise_variance   = .5;          
x                = linspace(-.5,.5,number_of_points)'; % x is between -.5 and .5
noise            = randn(number_of_points,1)*noise_variance^2;
y                = 3*x+12*x.^2+noise;                        % y is a quadratic function of x, the function we would like to recover
plot(x,y,'o')
% model:
model_degree = 3;
M = [];
for d = 0:model_degree
   M = [M x.^d]
end
% fit:
weights = M\y;
fit     = M*weights;
error   = sqrt(mean((fit-y).^2));
%
subplot(1,2,1)
plot(M,'linewidth',5)
box off
%
subplot(1,2,2)
plot(x,y,'o')
hold on
plot(x,fit,'k','linewidth',5)
box off
hold off;
%%
% fit a model with cross-validation 
K            = 5;
indices      = crossvalind('Kfold',y,K);
for k = 1:K
    test_indices  =  ismember(indices,k);
    train_indices = ~test_indices;
    
    weights    = M(train_indices,:)\y(train_indices);
    fit        = M(test_indices,:)*weights;
    
    test_error(k)    = sqrt(mean((fit-y(test_indices)).^2));
    train_error(k)   = sqrt(mean((fit-y(test_indices)).^2));
end
test_error  = mean(test_error);
train_error = mean(train_error);

%Why these two are different?
%%
number_of_points = 1000;
noise_variance   = 3;          
x                = linspace(-.5,.5,number_of_points)'; % x is between -.5 and .5
noise            = randn(number_of_points,1)*noise_variance^2;
y                = 3*x+1*x.^3+12*x.^2+noise;                        % y is a quadratic function of x, the function we would like to recover
for degree = 0:10 
    degree
    % model:
    M = make_my_model(x,degree);
    [test_err(degree+1),train_err(degree+1)] = fit_my_model(M,y);
end
plot(train_err,'ro-')
hold on
plot(test_err,'bo-')
hold off
%% Model complexity:
%
% Overfitting vs. underfitting: Learning too much from the training test. This happens when
% the model is too complex relative to the amount of data.


%% Classification
% Say you have a new device that detects infection in a blood sample. You
% tested on N labelled (infected, not infected) samples. For each sample
% your device returns 1 or 0. 
%
% How can we measure the performance of this device?
%
% Accuracy: The number of correct responses.
% Obviously it doesn't cover the full story. This brings us to the point
% where it is clear that we have to consider both labels and their
% respective outcomes.
%
% This is what a Confusion Matrix does:
%
%   A B
% A . .
% B . .
%
% a perfect classifier: has only diagonal non-zero entries.
%
% Precision of a classifier:
%
% Recall of a classifier
%
% Specificity: How good the classifier detects true examples (PRECISION).
% 
%
% Sensitivity: How good a classifier rejects a false example (RECALL)
% True positive rate.
%
%
% If the classifier returns always TRUE, it would have a perfect PRECISION,
% yet a very bad RECALL. The opposite holds for a classifier that always
% return FALSE.
%
%% The practical session:
%
% We will work with eye-movement data recorded while humans were looking at
% some photographic stimuli. Your task is to come up with an image-based feature that will
% classify image regions as fixated or not. We would like to quantify this
% model using area under the curve metric. I would like to think about
% upper and lower bounds of this model.
%


%
%% Let's illustrate how model complexity and number of data points influences training and testing errors.


%% 
%let's talk about this equation
%this is an equation of line
%1*x+2*y = 0
%this is the same as 
%[x y]*[1 2]' = 0;
% the pair of points that satisfy this equation form a line.
% for any x, there is one y. Because y has a coefficient of 2,
% for each x we have the negative half as y. 
% for example: if x = 2 then we have -1 for y so that the thing sums to 0.
% in general we have x = -2y
x = -5:.1:5;
w = [1 2 -2]
y = x*(-w(1)/w(2));

vector_to_line = @(w,x) x*(-w(1)/w(2));
% vector_to_line = @(w,x) (w(3)/w(1)) + x*(-w(1)/w(2));
% vector_to_line = @(w) linspace(-5,5,11)'*[-w(2) w(1)]/sqrt(w*w');
clf;
plot_vectors(w,'color','r')
hold on
plot(x,vector_to_line(w,x),'k','linewidth',3)
axis equal;set(gca,'xlim',[-5 5],'ylim',[-5 5])
%% Decision boundary
%we have now a line representation of a vector and we can use it as a
%decision boundary 
x              = linspace(-5,5,100)
w              = randn(1,2);
clf;
plot(data(:,1),vector_to_line(w,data(:,1)),'k','linewidth',4)
axis equal;set(gca,'xlim',[-5 5],'ylim',[-5 5])
hold on
pause(1)
%
[y,x]          = meshgrid(-5:.5:5,-5:.5:5)
data           = [x(:) y(:)];%each row is one imaginary data point.
for i = 1:size(data,1)
    X = data(i,:);
    %the sign of the projection determines the class of a datapoint.    
    %the value of w*X gives a signed measure of the distance from the
    %decision boundary.
    if w*X' > 0 %!!!!! The most important line
        plot(X(1),X(2),'ro');
    else
        plot(X(1),X(2),'bo');
    end
end
hold off;
h= legend('Decision Boundary','Class #1','Class #2')
set(h,'location','bestoutside')
% w determines the orientation of the decision boundary.
%
%w*X is the perpendicular distance to the decision boundary if w is has a
%unit length.
%% How do we come up with a decision boundary when we have observations from two classes A and B
%%
% We use more or less all the time the least-sqaure methdo to fit a line to
% a set of data points. 
% Error = Sum of all distances between a data point and the line.
% we find the line that minimizes this error term. The solution is
% analytically computed and unique.
% Now we have a slightly different situation where the error is not anymore
% a goodness of fit, but goodness of classification.
% 
C1 = randn(100,2)-2;
C2 = randn(100,2)+2;
clf;
plot(C1(:,1),C1(:,2),'ro')
hold on;
plot(C2(:,1),C2(:,2),'bo')
hold off;
axis equal
set(gca,'xlim',[-5 5],'ylim',[-5 5])
grid on;
% let create another matrix which identifies the class
X = [C1;C2]
T = [[ones(length(C1),1); -ones(length(C1),1)]]
% we aim to find the line classifies these two classes
%%
w = [0 1];
clf;
error = [];
for angle = 1:360
  
        alpha = deg2rad(angle);
        R     = [cos(alpha) sin(alpha);-cos(alpha) sin(alpha)]';
        w_rotated = R*w'
        subplot(1,2,1);
        plot(C1(:,1),C1(:,2),'ro')        
        hold on;
        plot(C2(:,1),C2(:,2),'bo')
        plot_vectors(w_rotated','color','r');
        hold on
        plot(-5:0.1:5,vector_to_line(w_rotated,-5:0.1:5),'k','linewidth',4)
        hold off
        axis equal;set(gca,'xlim',[-5 5],'ylim',[-5 5])            
        
        error(angle) = (X*w_rotated-T)'*(X*w_rotated-T);
        
        subplot(1,2,2);        
        plot(error,'o-')
        pause(0.1);drawnow;    
end

[~,i] = max(error)
alpha = deg2rad(i);
R     = [cos(alpha) sin(alpha);-cos(alpha) sin(alpha)]';
w_rotated = R*w'
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
w_analytical = T\X;
hold on;
plot_vectors(w_analytical,'color','r');
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
X = [C1;C2]
T = [[ones(length(C1),1); -ones(length(C2),1)]]
%
z = lsqlin([X ones(length(X),1)], T)
w_analytical = T\[X ones(length(X),1)];
hold on;
plot_vectors(w_analytical,'color','r');
hold on;
plot(-5:0.1:5,vector_to_line(w_analytical,-5:0.1:5),'k','linewidth',4)
hold off;
%% support vector machine:
%distance of a point to the line
vector_to_line = @(w,x) x*(-w(1)/w(2));
C1 = randn(50,2)+repmat([5 5],50,1);
C2 = randn(50,2)+repmat([-5 -5],50,1);
T  = [[ones(length(C1),1); -ones(length(C2),1)]]
X  = [C1;C2]
clf;
plot(C1(:,1),C1(:,2),'ro')
hold on;
plot(C2(:,1),C2(:,2),'bo')
hold off;
axis equal
set(gca,'xlim',[-10 10],'ylim',[-10 10])
grid on;
w = rand(1,2);
w = w/sqrt(w*w');
hold on;
plot(-5:0.1:5,vector_to_line(w,-5:0.1:5),'k','linewidth',2)
plot_vectors(w,'color','r');
hold on;
%
distances = w*X'.*T'
[min_distance,i] = min(distances);
plot(X(i,1),X(i,2),'+m','markersize',20)
%
min_distance
vector_to_line = @(w,x) (w(3)/w(1)) + x*(-w(1)/w(2));
plot(-5:0.1:5,vector_to_line([w min_distance],-5:0.1:5),'g-.','linewidth',2)

plot(-5:0.1:5,vector_to_line([w -min_distance],-5:0.1:5),'g-.','linewidth',2)
% clf
% plot(distances)
%% Let's compute the classification accuracy...
%
%
% The practical session:
%
% In this session, we will use eye-movement data recorded from humans when
% they were freely viewing pictures of faces. The aim of this project is to
% combine what we have learnt during day 1 and day2. Namely, I would like
% you to train an SVM classifier that will predict the identity of an
% individual based on the eye-movement patterns recorded during viewing of
% a face.

load eye_data.mat
%%
covmat    = cov(D');
fprintf('done\n')
fprintf('starting eigenvector computation\n')
[e dv]    = eig(covmat);
fprintf('done\n')
dv        = sort(diag(dv),'descend');
eigval = dv;
figure(100);
plot(cumsum(dv)./sum(dv),'o-');xlim([0 200]);drawnow
eigen     = fliplr(e);
%collect loadings of every trial
trialload = D'*eigen(:,1:teig)*diag(dv(1:teig))^-.5;%dewhitened
%%
sub_counter = 0;
result      = [];
w           = [];
random        = 0;
holdout_ratio = .5;
for sub1 = 1;unique(labels.sub)%go subject by subject    
    for sub2 = 2;unique(labels.sub)
        
        i = ismember(labels.sub,[sub1 sub2]);
        
        X    = trialload(i,:);
        Y    = labels.sub(i);
        
        svm = fitcsvm(X,Y(:),'crossval','off');
        cv  = crossval(svm)
        [a,score] = kfoldPredict(cv);
        
%         fitclinear(X,Y(:),'Holdout',holdout_ratio)
        
        
    end
end

    if random == 1
        warning('Randomizing labels as wanted. \n')
    end
    sub_counter = sub_counter + 1;
    ind_all     = ismember(labels.sub,sub);%this subject, this phase.
    %    
    
    
    for n = 1:tbootstrap%
        Ycond   = double(labels.cond(ind_all))';%labels of the fixation maps for this subject in this phase.
        X       = trialload(ind_all,:);%fixation maps of this subject in this phase.
        % now normal Holdout for every phase (which should all have the
        % same number of trials now)
        P       = cvpartition(Ycond,'Holdout',holdout_ratio); % divide training and test datasets respecting conditions
        i       = logical(P.training.*ismember(Ycond,[0 180]));%train using only the CS+ and CS? conditions.
        if random == 1
            
        else
            Mdl = fitclinear(X,Y,'Holdout',holdout_ratio)
            
            model   = svmtrain(Ycond(i), X(i,1:neig), '-t 0 -c 1 -q'); %t 0: linear, -c 1: criterion, -q: quiet
        end
        % get the hyperplane
        try
            w(:,sub_counter,n)          = model.SVs'*model.sv_coef;
        catch
            keyboard%sanity check: stop if something is wrong
        end
        %%
        cc=0;
        for cond = unique(Ycond)'
            cc                          = cc+1;
            i                           = logical(P.test.*ismember(Ycond,cond));%find all indices that were not used for training belonging to COND.
            [~, dummy]                  = evalc('svmpredict(Ycond(i), X(i,:), model);');%doing it like this supresses outputs.
            dummy                       = dummy == 0;%binarize: 1=CS+, 0=Not CS+
            result(cc,n,sub_counter)    = sum(dummy)/length(dummy);%get the percentage of CS+ responses for each CONDITION,BOOTSTR,SUBJECT
        end
    end
end