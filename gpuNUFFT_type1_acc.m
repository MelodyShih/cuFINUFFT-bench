%% Forward-Backward gradient test
clear all; 
close all; clear classes;

addpath(genpath('/mnt/home/yshih/gpuNUFFT/matlab/demo/utils'));
addpath(genpath('/mnt/home/yshih/gpuNUFFT/matlab/demo/../../gpuNUFFT'));

%% Data parameters
dim=3;
nupts_totry=[1];
tol=1e-2;
if (dim==2)
	N = 1000;
	M = 1e7;
end
if (dim==3)
	N = 100;
	M = 1e7;
end

if (dim==2)
	fileIDwrite = fopen("results/2d1_gpunufft_acc.dat", 'w');
else
	fileIDwrite = fopen("results/3d1_gpunufft_acc.dat", 'w');
end

fprintf(fileIDwrite, 'nupts code N1 N2 N3 M exec total totalgpumem cpuplan acc\n');
for dist=1:length(nupts_totry)
	N1 = N;
	N2 = N1;
	if dim==2
		N3 = 1;
		sizeCoo = [2 Inf];
		formatSpec = '(%f, %f)\n';
	else
		N3 = N1;
		sizeCoo = [3 Inf];
		formatSpec = '(%f, %f, %f)\n';
	end
	nupts = nupts_totry(dist); 
	fn1 = sprintf('true_%dd1_nupts%d_N%d_M1e7_4.bin', dim, nupts, N)
	fileID1 = fopen(fn1,'r');
	truef = fread(fileID1, 'float');
	truef = reshape(truef,2,[])';
	truef = truef(:,1)+1i*truef(:,2);
	
	fn = sprintf('data/DIM_%d_NUPTS_%d_N_%d_M_%d.dat', dim, nupts, N1, M)
	fileID = fopen(fn,'r');
	k = fscanf(fileID,formatSpec,sizeCoo);
	if dim==2
		truef = truef/sqrt(2*N1)/sqrt(2*N2);
	else
		truef = truef/sqrt(2*N1)/sqrt(2*N2)/sqrt(2*N3);
	end
	size(truef)
	for kw=1:2:12
		w = col(ones(size(k(1,:))));
		c = (1+1i)*ones(size(k,2),1);
		if (dim==2)
			[timeplan timetotalgem f] = gpunufft_2dtype1(dim, N1, N2, M, k, c, w, kw);
		else
			[timeplan timetotalgem f] = gpunufft_3dtype1(dim, N1, N2, N3, M, k, c, w, kw);
		end
		f = f(:);
	    acc = norm(f-truef)/norm(truef)
		fprintf(fileIDwrite, '%d %d %4d %4d %4d %10d %5.3e %5.3e %5.3e %5.3e %5.3e\n', ...
			nupts, 4, N1, N2, N3, M, 0, 0, timetotalgem, timeplan, acc);
	end
end

function [timeFTplan, timeFTH, f] = gpunufft_2dtype1(dim, N1, N2, M, k, c, w, kw)
	osf = 2;
	sw = 8;
	atomic = true;
	textures = true;
	loadbalancing = true;

	timeFTplan = 1000;
	timeFTH = 1000;

	for i=1:1
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
end

function [timeFTplan, timeFTH, f] = gpunufft_3dtype1(dim, N1, N2, N3, M, k, c, w, kw)
	osf = 2;
	sw = 8;
	atomic = true;
	textures = true;
	loadbalancing = true;

	timeFTplan = 1000;
	timeFTH = 1000;

	for i=1:3
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
end
