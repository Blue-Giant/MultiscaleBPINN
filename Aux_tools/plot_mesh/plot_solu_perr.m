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


data2solu_train = load('solu2train.mat');
data2force_train = load('force2train.mat');

XY_train2solu = data2solu_train.xu_train;
solu2train = data2solu_train.Utrain_sin;
X_train2solu = XY_train2solu(:, 1);
Y_train2solu = XY_train2solu(:, 2);

data2solu_test = load('solu2test.mat');

figure('name','true')
utrue = data2solu_test.Uexact_sin;
mesh_true = plot_fun2in(geom,utrue);
hold on
scatter3(X_train2solu, Y_train2solu, solu2train, 50, 'k*');
hold on

figure('name','Mean_Predict')
Umean = data2solu_test.Umean_sin;
mesh_Umean = plot_fun2in(geom,Umean);
hold on

point_abs = abs(utrue - Umean);
figure('name','Point ABS')
mesh_PABS = plot_fun2in(geom,point_abs);
hold on



