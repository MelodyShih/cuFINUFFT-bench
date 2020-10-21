%% Forward-Backward gradient test
clear all; 
close all; clear classes;

addpath(genpath('/mnt/home/yshih/gpuNUFFT/matlab/demo/utils'));
addpath(genpath('/mnt/home/yshih/gpuNUFFT/matlab/demo/../../gpuNUFFT'));

%% Data parameters
dim=3;
nupts_totry=[1,2,3];
tol=1e-6;
if (dim==2)
	N_totry=[64, 128, 256, 512, 1024, 2048];
	density_totry=[0.1,1,10];
end
if (dim==3)
	N_totry=[32, 64, 128, 256];
	density_totry=[0.1,1];
end

if (dim==2)
	fileIDwrite = fopen("results/2d2_gpunufft.dat", 'w');
else
	fileIDwrite = fopen("results/3d2_gpunufft.dat", 'w');
end
fprintf(fileIDwrite, 'nupts code den N1 N2 N3 M tol exec total totalgpumem cpuplan');
for dist=1:length(nupts_totry)
	for i=1:length(N_totry)
		for d=1:length(density_totry)
			N1 = N_totry(i);
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
			density=density_totry(d);
			M = floor(N1*N2*N3*2^dim*density);
			nupts = nupts_totry(dist); 
			fn = sprintf('data/DIM_%d_NUPTS_%d_N_%d_M_%d.dat', dim, nupts, N1, M);
			fileID = fopen(fn,'r');
			k = fscanf(fileID,formatSpec,sizeCoo);
			w = col(ones(size(k(1,:))));
			c = (1+1i)*ones([N1, N2, N3]);
			if (dim==2)
				[timeplan timetotalgem] = gpunufft_2dtype2(dim, N1, N2, M, k, c, w);
			else
				[timeplan timetotalgem] = gpunufft_3dtype2(dim, N1, N2, N3, M, k, c, w);
			end
			fprintf(fileIDwrite, '%d %d %3.1e %4d %4d %4d %10d %5.3e %5.3e %5.3e %5.3e %5.3e\n', ...
				nupts, 4, density, N1, N2, N3, M, tol, 0, 0, timetotalgem, timeplan);
		end
	end
end

%isign = 1;
%x = k(1,:)';
%y = k(2,:)';
%nt1 = floor(0.45*N1); nt2 = floor(0.35*N2); nt3=floor(0.12*N3);             % pick mode indices
%fe = sum(c.*exp(1i*2*pi*isign*(nt1*x+nt2*y+nt3*z)))/sqrt(2*N1)/sqrt(2*N2);                 % exact
%of1 = floor(N1/2)+1; of2 = floor(N2/2)+1; of3=floor(N3/2)+1;   
%fprintf('2D type-1: rel err in F[%d,%d] is %.3g\n',nt1,nt2,abs((fe-f(nt1+of1,nt2+of2,nt3+of3))/max(f(:))))

function [timeFTplan, timeFTH] = gpunufft_2dtype2(dim, N1, N2, M, k, f, w)
	osf = 2;
	kw = 8; %1 also possible (nearest neighbor) 
	sw = 8;
	atomic = true;
	textures = true;
	loadbalancing = true;

	timeFTplan = 1000;
	timeFTH = 1000;

	for i=1:5
		tic
		FT = gpuNUFFT(k,w,osf,kw,sw,[N1,N2],[],atomic,textures,loadbalancing);
		t = toc;
		timeFTplan = min(timeFTplan, t);
		%disp(['Time plan: ', num2str(timeFTplan), ' s']);

		tic
		c = FT*f;
		t = toc;
		timeFTH = min(timeFTH, t);
		%disp(['Time adjoint: ', num2str(timeFTH), ' s']);
		%fprintf('               %.3e NUpts/s\n', M/timeFTH)
	end
end

function [timeFTplan, timeFTH] = gpunufft_3dtype2(dim, N1, N2, N3, M, k, f, w)
	osf = 2;
	kw = 8; %1 also possible (nearest neighbor) 
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
		c = FT*f;
		t = toc;
		timeFTH = min(timeFTH, t);
		%disp(['Time adjoint: ', num2str(timeFTH), ' s']);
		%fprintf('               %.3e NUpts/s\n', M/timeFTH)
	end
end
