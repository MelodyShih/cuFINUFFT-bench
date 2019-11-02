#include <cuComplex.h>

#ifndef FLT
typedef float FLT;
#endif
#ifndef IMA 
#define IMA std::complex<FLT>(0.0,1.0) 
#endif
#ifndef CUCPX
	#ifdef GPU
		typedef cuFloatComplex CUCPX;
	#else
		typedef std::complex<float> CUCPX;
	#endif
#endif

#ifndef rand01
#define rand01() ((FLT)rand()/RAND_MAX)
#endif
#ifndef randm11
#define randm11() (2*rand01() - (FLT)1.0)
#endif
#ifndef crandm11
#define crandm11() (randm11() + IMA*randm11())
#endif

/* 
 * Create data for Type 1 transformation (nonuniform -> uniform) 
 *
 * 	c[i]   : strength of ith nonuniform pt
 * 	x[i*ix]: x coordinate of ith nonuniform pt
 * 	y[i*iy]: y coordinate of ith nonuniform pt
 * 	z[i*iz]: z coordinate of ith nonuniform pt
 * 	ix, iy, iz: the gap between ith and (i+1)th nonuniform pt data
 *
 */
void create_data_type1(int nupts_distr, int dim, int M, FLT* x, FLT* y, FLT* z, 
	int ix, int iy, int iz, CUCPX* c, FLT scale)
{
	int i;
	switch(nupts_distr){
		case 1:
			{
				for (i = 0; i < M; i++) {
					x[i*ix] = scale*randm11();
					if(dim > 1)
						y[i*iy] = scale*randm11();
					if(dim > 2)
						z[i*iz] = scale*randm11();
#ifdef GPU
					c[i].x = randm11();
					c[i].y = randm11();
#else
					c[i] = crandm11();
#endif
				}
			}
			break;
		default:
			fprintf(stderr,"Invalid distribution of nonuniform points\n");
	}	
}
#if 0
void create_data_type2(int nupts_distr, cunfft_plan *plan, int dim, int N1, int N2, 
	int N3)
{
	switch(nupts_distr){
		case 1:
			{
				for (int i = 0; i < plan->M_total; i++) {
					plan->x[dim*i] = 0.5*randm11(); // x in [-pi,pi)
					if(dim > 1)
						plan->x[dim*i+1] = 0.5*randm11();
					if(dim > 2)
						plan->x[dim*i+2] = 0.5*randm11();
				}
			}
			break;
		case 2:
			{
				for (int i = 0; i < plan->M_total; i++) {
					plan->x[dim*i] = 0.5*rand01()/(N1*2*2/32); // x in [-pi,pi)
					if(dim > 1)
						plan->x[dim*i+1] = 0.5*rand01()/(N2*2*2/32);
					if(dim > 2)
						plan->x[dim*i+2] = 0.5*rand01()/(N3*2*2/32);
				}
			}
			break;
		default:
			cerr<<"Invalid distribution of nonuniform points"<<endl;
	}
	for(int i=0; i<plan->n_total; i++){
		plan->g[i].x = randm11();
		plan->g[i].y = randm11();
	}
}
#endif
