clc;
clear all
close all

q = 7;
nl = 2;
T = [];
J = [];

% geom: (only square geometry available now)
% generating 2d square mesh for the region [-1, 1] x [-1 1]
geom.q = q;
geom.nl = nl;
geom.L = 2; % side length 
geom.dim = 2; % dimension of the problem
geom.m = 2^geom.dim; % 
geom.N1 = 2^q; % dofs in one dimension
geom.N = (geom.m)^geom.q; % dofs in the domain
geom.h = geom.L/(geom.N1+1); % grid size
geom.xstart = -1;
geom.xend = 1;
geom.ystart = -1;
geom.yend = 1;

geom = assemble_fmesh(geom);

data2force_train = load('force2train.mat');

XY_train2force = data2force_train.xf_train;
force2train = data2force_train.Ftrain_sin;
X_train2force = XY_train2force(:, 1);
Y_train2force = XY_train2force(:, 2);

data2force_test = load('force2test.mat');

figure('name','true')
utrue = data2force_test.Fexact_sin;
mesh_true = plot_fun2in(geom,utrue);
hold on
scatter3(X_train2force, Y_train2force, force2train, 40, 'r*');
hold on

figure('name','Mean_Predict')
Umean = data2force_test.Fmean_sin;
mesh_Umean = plot_fun2in(geom,Umean);
hold on

point_abs = abs(utrue - Umean);
figure('name','Point ABS')
mesh_PABS = plot_fun2in(geom,point_abs);
hold on



