%% read data:
%
% Read face images from the database and extract facial expression from the
% file name.
%
%
%
%Reference:
%Lundqvist, D., Flykt, A., & ?hman, A. (1998). The Karolinska Directed
%Emotional Faces - KDEF, CD ROM from Department of Clinical Neuroscience,
%Psychology section, Karolinska Institutet, ISBN 91-630-7164-9.   


cd /Users/onat/Desktop/MachineLearning/face_database/
f                = dir('*.JPG');                        % all file names in the database directory
number_of_images = length(f);                           % number of face images
im               = [];                                  % initialize the variable that will contain images

emotion_matrix   = zeros(number_of_images,5);           % variable to store emotion values 
emotion_codes    = ... 
{'AF'    'AN'    'DI'    'HA'    'NE'    'SA'    'SU'}; %possible emotions:
%afraid, angry, disgust, happy, neutral, sad, surprised


for i= 1:number_of_images
    fprintf('Reading file %s\n',f(i).name)
    im(:,:,i)     = imresize(rgb2gray(imread(f(i).name)),.1);
%     im(:,:,:,i)     = imread(f(i).name);
    % extract the emotion code from file name
    emotion_code{i} = f(i).name([end-6:end-5]);   
end
image_width    = size(im,2);
image_height   = size(im,1);
im             = reshape(im,image_width*image_height,number_of_images)';
im             = demean(im);
emotional_code = dummyvar(categorical(emotion_code(:)));

save /Users/onat/Desktop/MachineLearning/face_database.mat emotional_code im image_width image_height