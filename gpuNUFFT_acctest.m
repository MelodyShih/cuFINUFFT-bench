%% Forward-Backward gradient test
clear all; 
close all; clear classes;

addpath(genpath('/mnt/home/yshih/gpuNUFFT/matlab/demo/utils'));
addpath(genpath('/mnt/home/yshih/gpuNUFFT/matlab/demo/../../gpuNUFFT'));

%% Data parameters
%sizeCoo = [2 Inf];
%fileID = fopen('coo.dat','r');
%formatSpec = '(%f, %f)\n';
%k = fscanf(fileID,formatSpec,sizeCoo);

N1=1024;
N2=1024;
N3=1;
dim=2;
M = N1*N2;
k = rand([2,M])-0.5;
w = col(ones(size(k(1,:))));
c = (1+1i)*ones(size(k,2),1);
[timeplan timetotalgem f] = gpunufft_2dtype1(dim, N1, N2, M, k, c, w);

isign = 1;
x = k(1,:)';
y = k(2,:)';
nt1 = floor(0.45*N1); nt2 = floor(0.35*N2);             % pick mode indices
fe = sum(c.*exp(1i*2*pi*isign*(nt1*x+nt2*y)))/sqrt(2*N1)/sqrt(2*N2);                 % exact
of1 = floor(N1/2)+1; of2 = floor(N2/2)+1;
fe
f(nt1+of2, nt2+of2)
fprintf('2D type-1: rel err in F[%d,%d] is %.3g\n',nt1,nt2,abs((fe-f(nt1+of1,nt2+of2))/max(f(:))))
fprintf('2D type-1: abs err in F[%d,%d] is %.3g\n',nt1,nt2,abs((fe-f(nt1+of1,nt2+of2))))

isign = -1;
N1=1024;
N2=1024;
N3=1;
dim=2;
M = N1*N2*N3;
k = rand([2,M])-0.5;
x = k(1,:)';
y = k(2,:)';
w = col(ones(size(k(1,:))));
f = (1+1i)*(ones(N1,N2));
j = ceil(0.93*M);
[mm1,mm2] = ndgrid(ceil(-N1/2):floor((N1-1)/2),ceil(-N2/2):floor((N2-1)/2));
ce = sum(f(:).*exp(1i*isign*2*pi*(mm1(:)*x(j)+mm2(:)*y(j))))/sqrt(2*N1)/sqrt(2*N2);
[timeplan timetotalgem c] = gpunufft_2dtype2(dim, N1, N2, M, k, f, w);
ce
c(j)
fprintf('2D type-2: rel err in c[%d] is %.3g\n',j,abs((ce-c(j))/max(c(:))))
fprintf('2D type-2: abs err in c[%d] is %.3g\n',j,abs((ce-c(j))))


function [timeFTplan, timeFTH, f] = gpunufft_2dtype1(dim, N1, N2, M, k, c, w)
	osf = 2;
	kw = 8; %1 also possible (nearest neighbor) 
	sw = 8;
	atomic = true;
	textures = true;
	loadbalancing = true;

	timeFTplan = 1000;
	timeFTH = 1000;

	tic
	FT = gpuNUFFT(k,w,osf,kw,sw,[N1,N2],[],atomic,textures,loadbalancing);
	t = toc;
	timeFTplan = min(timeFTplan, t);
	disp(['Time plan: ', num2str(timeFTplan), ' s']);

	tic
	f = (FT'*c);
	t = toc;
	timeFTH = min(timeFTH, t);
	disp(['Time adjoint: ', num2str(timeFTH), ' s']);
	fprintf('               %.3e NUpts/s\n', M/timeFTH)
end

function [timeFTplan, timeFTH, c] = gpunufft_2dtype2(dim, N1, N2, M, k, f, w)
	osf = 2;
	kw = 8; %1 also possible (nearest neighbor) 
	sw = 8;
	atomic = true;
	textures = true;
	loadbalancing = true;

	timeFTplan = 1000;
	timeFTH = 1000;

	tic
	FT = gpuNUFFT(k,w,osf,kw,sw,[N1,N2],[],atomic,textures,loadbalancing);
	t = toc;
	timeFTplan = min(timeFTplan, t);
	%disp(['Time plan: ', num2str(timeFTplan), ' s']);

	tic
	c = FT*(f);
	t = toc;
	timeFTH = min(timeFTH, t);
	%disp(['Time adjoint: ', num2str(timeFTH), ' s']);
	%fprintf('               %.3e NUpts/s\n', M/timeFTH)
end
