%% Forward-Backward gradient test
clear all; 
close all; clear classes;

addpath(genpath('/mnt/home/yshih/gpuNUFFT/matlab/demo/utils'));
addpath(genpath('/mnt/home/yshih/gpuNUFFT/matlab/demo/../../gpuNUFFT'));

% Data parameters
%sizeCoo = [2 Inf];
%fileID = fopen('coo.dat','r');
%formatSpec = '(%f, %f)\n';
%k = fscanf(fileID,formatSpec,sizeCoo);

% 2D
N1=128;
N2=128;
N3=1;
dim=2;

% data from file
sizeCoo = [2 Inf];
%fileID = fopen('data/DIM_2_NUPTS_1_N_32_M_1024.dat','r');
%fileIDwrite = fopen('results/acc_2d_32_gpunufft.dat', 'w');
fileID = fopen('data/DIM_2_NUPTS_1_N_128_M_16384.dat','r');
fileIDwrite = fopen('results/acc_2d_128_gpunufft.dat', 'w');
formatSpec = '(%f, %f)\n';
k = fscanf(fileID,formatSpec,sizeCoo);
M = size(k,2);
w = col(ones(size(k(1,:))));
c = (1+1i)*ones(size(k,2),1);

% rand data
%M = N1*N2*N3;
%k = rand([2,M])-0.5;

% 2D type 1
isign = 1;
x = k(1,:)';
y = k(2,:)';
nt1 = floor(0.45*N1); nt2 = floor(0.35*N2); % pick mode indices
fe = sum(c.*exp(1i*2*pi*isign*(nt1*x+nt2*y)))/sqrt(2*N1)/sqrt(2*N2); % exact
of1 = floor(N1/2)+1; of2 = floor(N2/2)+1;

n1modes = -N1/2:N1/2-1;
n2modes = -N2/2:N2/2-1;
[n1allmodes, n2allmodes] = ndgrid(n1modes, n2modes);
Fe = exp(1i*2*pi*isign*(n1allmodes(:)*x'+n2allmodes(:)*y'))*c/sqrt(2*N1)/sqrt(2*N2);
Fe = reshape(Fe, N1, N2);

fprintf('lib type N1 N2 N3 M w reltol\n');
for kw=3:2:7
	[timeplan timetotalgem fk] = gpunufft_2dtype1(dim, N1, N2, M, k, c, w, kw);
	fprintf('[w=%2d] 2D type-1: rel err in F[%d,%d] is %.3g\n',kw,nt1,nt2, ...
            abs((fe-fk(nt1+of1,nt2+of2))/max(fk(:))))
	fprintf('[w=%2d] 2D type-1: abs err in F[%d,%d] is %.3g\n',kw,nt1,nt2,...
            abs((fe-fk(nt1+of1,nt2+of2))))
	err = norm(Fe(:) - fk(:))/norm(Fe(:));
	fprintf('[w=%2d] [acc check] direct calculation: rel l2-err of result F is %.3g\n',err);
	fprintf(fileIDwrite, '%d %d %d %d %d %d %d %f\n',4, 1, N1, N2, N3, M, kw, err);
end

% 2D type 2
isign = -1;
w = col(ones(size(k(1,:))));
f = (1+1i)*(ones(N1,N2));
j = ceil(0.93*M);
[mm1,mm2] = ndgrid(ceil(-N1/2):floor((N1-1)/2),ceil(-N2/2):floor((N2-1)/2));
ce = sum(f(:).*exp(1i*isign*2*pi*(mm1(:)*x(j)+mm2(:)*y(j))))/sqrt(2*N1)/sqrt(2*N2);
Ce = zeros(M,1);
for i=1:M
Ce(i) = sum(f(:).*exp(1i*isign*2*pi*(mm1(:)*x(i)+mm2(:)*y(i))))/sqrt(2*N1)/sqrt(2*N2);
end
for kw=3:2:7
	[timeplan timetotalgem c] = gpunufft_2dtype2(dim, N1, N2, M, k, f, w, kw);
	fprintf('[w=%2d] 2D type-2: rel err in c[%d] is %.3g\n',kw, j,abs((ce-c(j))/max(c(:))))
	fprintf('[w=%2d] 2D type-2: abs err in c[%d] is %.3g\n',kw, j,abs((ce-c(j))))
	err  = norm(Ce(:) - c(:))/norm(Ce(:));
	fprintf('[w=%2d] [acc check] direct calculation: rel l2-err of result C is %.3g\n',kw, err);
	fprintf(fileIDwrite, '%d %d %d %d %d %d %d %f\n',4, 2, N1, N2, N3, M, kw, err);
end
clear all; 
close all; clear classes;

addpath(genpath('/mnt/home/yshih/gpuNUFFT/matlab/demo/utils'));
addpath(genpath('/mnt/home/yshih/gpuNUFFT/matlab/demo/../../gpuNUFFT'));

% 3D
N1=16;
N2=16;
N3=16;
dim=3;
sizeCoo = [3 Inf];
fileID = fopen('data/DIM_3_NUPTS_1_N_16_M_4096.dat','r');
fileIDwrite = fopen('results/acc_3d_16_gpunufft.dat', 'w');
%fileID = fopen('data/DIM_3_NUPTS_1_N_8_M_512.dat','r');
%fileIDwrite = fopen('results/acc_3d_8_gpunufft.dat', 'w');
formatSpec = '(%f, %f, %f)\n';
k = fscanf(fileID,formatSpec,sizeCoo);
M = size(k,2);

% rand data
%M = N1*N2*N3;
%k = rand([3,M])-0.5;
isign = 1;
w = col(ones(size(k(1,:))));
c = (1+1i)*ones(size(k,2),1);
x = k(1,:)';
y = k(2,:)';
z = k(3,:)';
nt1 = floor(0.45*N1); nt2 = floor(0.35*N2); nt3 = floor(0.13*N3);% pick mode indices
fe = sum(c.*exp(1i*2*pi*isign*(nt1*x+nt2*y+nt3*z)))/sqrt(2*N1)/sqrt(2*N2); % exact
of1 = floor(N1/2)+1; of2 = floor(N2/2)+1; of2 = floor(N2/2)+1;

n1modes = -N1/2:N1/2-1;
n2modes = -N2/2:N2/2-1;
n3modes = -N3/2:N3/2-1;
[n1allmodes, n2allmodes, n3allmodes] = ndgrid(n1modes, n2modes, n3modes);
Fe = exp(1i*2*pi*isign*(n1allmodes(:)*x'+n2allmodes(:)*y'+n3allmodes(:)*z'))*c...
     /sqrt(2*N1)/sqrt(2*N2)/sqrt(2*N3);
Fe = reshape(Fe, N1, N2, N3);

fprintf('lib type N1 N2 N3 M w reltol\n');
for kw=3:2:7
	[timeplan timetotalgem fk] = gpunufft_3dtype1(dim, N1, N2, N3, M, k, c, w, kw);
	fprintf('[w=%2d] 3D type-1: rel err in F[%d,%d,%d] is %.3g\n',kw,nt1,nt2,nt3, ...
            abs((fe-fk(nt1+of1,nt2+of2))/max(fk(:))))
	fprintf('[w=%2d] 3D type-1: abs err in F[%d,%d,%d] is %.3g\n',kw,nt1,nt2,nt3,...
            abs((fe-fk(nt1+of1,nt2+of2))))
	err = norm(Fe(:) - fk(:))/norm(Fe(:));
	fprintf('[w=%2d] [acc check] direct calculation: rel l2-err of result F is %.3g\n',err);
	fprintf(fileIDwrite, '%d %d %d %d %d %d %d %f\n',4, 1, N1, N2, N3, M, kw, err);
end

% 3D type 2
isign = -1;
w = col(ones(size(k(1,:))));
f = (1+1i)*(ones(N1,N2,N3));
j = ceil(0.93*M);
[mm1,mm2,mm3] = ndgrid(ceil(-N1/2):floor((N1-1)/2), ...
                       ceil(-N2/2):floor((N2-1)/2), ...
                       ceil(-N3/2):floor((N3-1)/2));
ce = sum(f(:).*exp(1i*isign*2*pi*(mm1(:)*x(j)+mm2(:)*y(j)+mm3(:)*z(j)))) ...
     /sqrt(2*N1)/sqrt(2*N2)/sqrt(2*N3);
Ce = zeros(M,1);
for i=1:M
Ce(i) = sum(f(:).*exp(1i*isign*2*pi*(mm1(:)*x(i)+mm2(:)*y(i)+mm3(:)*z(i))))...
        /sqrt(2*N1)/sqrt(2*N2)/sqrt(2*N3);
end

for kw=3:2:7 % test with different kernel width
	[timeplan timetotalgem c] = gpunufft_3dtype2(dim, N1, N2, N3, M, k, f, w, kw);
	fprintf('[w=%2d] 3D type-2: rel err in c[%d] is %.3g\n',kw, j,...
            abs((ce-c(j))/max(c(:))))
	fprintf('[w=%2d] 3D type-2: abs err in c[%d] is %.3g\n',kw, j,abs((ce-c(j))))
	err  = norm(Ce(:) - c(:))/norm(Ce(:));
	fprintf('[w=%2d] [acc check] direct calculation: rel l2-err of result C is %.3g\n',...
kw, err);
	fprintf(fileIDwrite, '%d %d %d %d %d %d %d %f\n',4, 2, N1, N2, N3, M, kw, err);
end
clear all; 


function [timeFTplan, timeFTH, f] = gpunufft_2dtype1(dim, N1, N2, M, k, c, w, kw)
	osf = 2;
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
	f = (FT'*c);
	t = toc;
	timeFTH = min(timeFTH, t);
	%disp(['Time adjoint: ', num2str(timeFTH), ' s']);
	%fprintf('               %.3e NUpts/s\n', M/timeFTH)
end

function [timeFTplan, timeFTH, f] = gpunufft_3dtype1(dim, N1, N2, N3, M, k, c, w, kw)
	osf = 2;
	sw = 8;
	atomic = true;
	textures = true;
	loadbalancing = true;

	timeFTplan = 1000;
	timeFTH = 1000;

	tic
	FT = gpuNUFFT(k,w,osf,kw,sw,[N1,N2,N3],[],atomic,textures,loadbalancing);
	t = toc;
	timeFTplan = min(timeFTplan, t);
	%disp(['Time plan: ', num2str(timeFTplan), ' s']);

	tic
	f = (FT'*c);
	t = toc;
	timeFTH = min(timeFTH, t);
	%disp(['Time adjoint: ', num2str(timeFTH), ' s']);
	%fprintf('               %.3e NUpts/s\n', M/timeFTH)
end

function [timeFTplan, timeFTH, c] = gpunufft_2dtype2(dim, N1, N2, M, k, f, w, kw)
	osf = 2;
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

function [timeFTplan, timeFTH, c] = gpunufft_3dtype2(dim, N1, N2, N3, M, k, f, w, kw)
	osf = 2;
	sw = 8;
	atomic = true;
	textures = true;
	loadbalancing = true;

	timeFTplan = 1000;
	timeFTH = 1000;

	tic
	FT = gpuNUFFT(k,w,osf,kw,sw,[N1,N2,N3],[],atomic,textures,loadbalancing);
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
