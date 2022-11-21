%% Initialize the procesure
clear all; 
close all; 
clc;

%% Load the data
[feature1, feature2, feature3, feature4, class] = textread('IrisData.txt', '%f%f%f%f%s', 'delimiter', ',');
feature = [feature1'; feature2'; feature3'; feature4']';
% normalize data in [0,1]
[m, n] = size(feature1);
x_max=max(feature);
x_min=min(feature);
feature=(feature-ones(m,1)*x_min)./(ones(m,1)*(x_max-x_min));

target = zeros(m, 1);
target(strcmp(class, 'Iris-setosa')) = 1;
target(strcmp(class, 'Iris-versicolor')) = 2;
target(strcmp(class, 'Iris-virginica')) = 3;
label = {'setosa','versicolor','virginica'};

%% Split the data into 70% training data and 30% validation data
total_data = [feature'; target']';
[train_data, val_data] = split2train_test(total_data,0.7);
train_x = train_data(:,1:4);
train_y = train_data(:,5);
train_y = class2label(train_y);
val_x = val_data(:,1:4);
val_y = val_data(:,5);
%val_y = class2label(val_y);

%% create neutral networks
net = newff(train_x', train_y', [4,5,3]);
% set the neutral networks parameters
net.trainParam.epochs = 1000;
net.trainParam.lr = 0.01;
% stochastic gradient descent
net.trainFcn = 'traingdx';
net = train(net, train_x', train_y');
%save the mdoel
save filename net;

%% test the neutral networks
an = sim(net, val_x');
an = label2class(an');

confusionmat =confusionmat (val_y',an');
confusionchart(confusionmat,label);

function onehot = class2label(class)
%% This fucntion changes [3,2,1,1] to 
%% [0,0,1;0,1,0;1,0,0;1,0,0]
n = length(unique(class'));
onehot = full(ind2vec(class',n));
onehot = onehot';
end

function class = label2class(label)
%% This fucntion changes [0,0,1;0,1,0;1,0,0;1,0,0] to 
%% [3,2,1,1]
label = round(label);
class = vec2ind(label');
class = class';
end


function [train, test] = split2train_test(input,proportion)
%% This fucntion randomly split data into train and valadation(test) set
 
% input parameters：
% input : original dataset, including features and target
% proportion: the ratio of the train data
 
% output parameters：
% train:train set
% test:valadation(test) set
 
rows=size(input,1);
split=randindex(rows,proportion);
train=input(split==0,:);
test=input(split==1,:);
end
 
function randindex=randindex(n,proportion)
%% This function return the random split \
%% index of train and valadation(test) set
    randindex=zeros(n,1);
    rng('default'); % fixed random seed
    for i=1:n
       if rand(1)>proportion
           randindex(i)=1;
       end
    end
end

