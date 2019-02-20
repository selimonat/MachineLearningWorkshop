function [images,emotion_code]=read_preprocess_faces
%% read data:
%
% Read face images from the database folder, preprocess them and extract
% facial expression from the file name. You may want to change the path
% variable below.
%
%
%
% Reference:
% Lundqvist, D., Flykt, A., & ?hman, A. (1998). The Karolinska Directed
% Emotional Faces - KDEF, CD ROM from Department of Clinical Neuroscience,
% Psychology section, Karolinska Institutet, ISBN 91-630-7164-9.   

%%
path_to_faces =  '~/Desktop/MachineLearning/face_database/';
cd(path_to_faces);
f                = dir('*.JPG');                        % all file names in the database directory
number_of_images = length(f);                           % number of face images
images           = [];                                  % initialize the variable that will contain images

emotion_codes    = ... 
{'AF'    'AN'    'DI'    'HA'    'NE'    'SA'    'SU'}; % possible emotions:
%afraid, angry, disgust, happy, neutral, sad, surprised

% run through images
for i = 1:number_of_images
    fprintf('Reading file %s (%d,%d)\n',f(i).name,i,number_of_images)    
    % extract the emotion code from file name, store the expression
    emotion_code{i} = f(i).name([end-6:end-5]);    
    % read the current image
    current_image   = imread(f(i).name);    
    % Code your preprocessing steps here:
    % Apply your preprocessing steps to the CURRENT_IMAGE and store the
    % result in IMAGES matrix below.
%     images(:,:,:,i)   = current_image;    
end
% Return emotional_code as a matrix where each column is one expression and
% each row is one image 
emotion_code = dummyvar(categorical(emotion_code(:)));