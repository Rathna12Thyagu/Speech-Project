#include "neuralNets.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EULER_NUMBER 2.71828
#define MAX_LIMIT 10

double** generate2d(int x, int y, double fan_in, double fan_out) {
    double limit = sqrt(6.0 / (fan_in));
    double** mat = (double**)malloc(x * sizeof(double*));
    for (int i = 0; i < x; i++) {
        mat[i] = (double*)malloc(y * sizeof(double));
        for (int j = 0; j < y; j++) {
            mat[i][j] = ((double)rand() / RAND_MAX) * 2 * limit - limit;
        }
    }
    return mat;
}

void prop(double** w,double**m,double**b,int x,int y,int z,double** out)
{
	//printf("%s\n","forward Propagation");
	for(int i=0;i<x;i++)
	{
		for(int j=0;j<y;j++)
		{
			for(int k=0;k<z;k++)
			{
				out[i][j] += w[i][k] * m[k][j];
			}
		}
	}
}
void relu(double** a, int x, int y, double** out) {
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            out[i][j] = fmax(0, a[i][j]);
        }
    }
}

void relu_derivative(double** a, int x, int y, double** out) {
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            out[i][j] = (a[i][j] > 0) ? 1 : 0;
        }
    }
}
double** oneHot(int* a,int x)
{
	double** out = (double**)malloc(x*sizeof(double*));
	for(int i=0;i<x;i++)
	{
		out[i] = (double*)malloc(7*sizeof(double));
	}
	for(int i=0;i<x;i++)
	{
		for(int j=0;j<7;j++)
		{
			out[i][j] = 0;
		}
	}
	for(int i=0;i<x;i++)
	{
		out[i][a[i]] = 1.0;
	}
	return out;
}
void transpose(double** m,int x,int y,double** out)
{
	for(int i=0;i<y;i++)
	{
		for(int j=0;j<x;j++)
		{
			out[i][j] = m[j][i];
		}
	}
}
void backprop_weights(double** a,double** b,int x,int y,int z,double** out)
{
	for(int i=0;i<x;i++)
	{
		for(int j=0;j<y;j++)
		{
			out[i][j] = 0.0;
		}
	}
	double t = 1.0/7000.0;
	//printf("%s\n","Backprop Weights");
	for(int i=0;i<x;i++)
	{
		for(int j=0;j<y;j++)
		{
			for(int k=0;k<z;k++)
			{
				out[i][j] += t*a[i][k]*b[k][j];
			}
		}
	}
}
void backprop_bias(double** m,int x,int y,int z,double** out)
{
	double t = 1.0/7000.0;
	for(int i=0;i<x;i++)
	{
		for(int j=0;j<z;j++)
		{
			out[i][0] += t*m[i][j];
		}
	}
}
void backprop_error(double** a,double** b,int x,int y,int z,double**c,double** out)
{
	//printf("%s\n","Error_prop");
	for(int i=0;i<x;i++)
	{
		for(int j=0;j<y;j++)
		{
			for(int k=0;k<z;k++)
			{
				out[i][j] += a[i][k] * b[k][j];
			}
		}
	}
	for(int i=0;i<x;i++)
	{
		for(int j=0;j<y;j++)
		{
			out[i][j] = out[i][j]*c[i][j];
		}
	}
}
void update(double** a,double**b,int x,int y)
{
	for(int i=0;i<x;i++)
	{
		for(int j=0;j<y;j++)
		{
			a[i][j] = a[i][j] - 0.01*b[i][j];
		}
	}
}
void get_predictions(double** a,int x,int y,int* out)
{
	for(int i=0;i<y;i++)
	{
		double max = 0;int ind = 0;
		for(int j=0;j<x;j++)
		{
			if(a[j][i]>max)
			{
				max = a[j][i];
				ind = j;
			}
			else
			{continue;}
		}
		//printf("%f--->%d\n",max,ind);
		out[i] = ind;
	}
}
double accuracy(int* a,int* b)
{
	double c=0;
	for(int i=0;i<7000;i++)
	{
		if(a[i]==b[i])
		{
			c = c + 1;
		}
	}
	printf("Hit : %f\n",c);
	return c/7000;
}
double** allocateMem(int x,int y)
{
	double** m = (double**)malloc(x*sizeof(double*));
	for(int i=0;i<x;i++)
		m[i] = (double*)malloc(y*sizeof(double));
	for(int i=0;i<x;i++)
	{
		for(int j=0;j<y;j++)
		{
			m[i][j] = 0.0;
		}
	}
	return m;
}
void reset(int x,int y,double** out)
{
	for(int i=0;i<x;i++)
		for(int j=0;j<y;j++)
			out[i][j] = 0.0;
}
void softmax(double** a, int x, int y, double** out) {
    for (int j = 0; j < y; j++) {
        double max_val = a[0][j];
        for (int i = 1; i < x; i++) {
            if (a[i][j] > max_val) {
                max_val = a[i][j];
            }
        }
        double sum = 0.0;
        for (int i = 0; i < x; i++) {
            a[i][j] = exp(a[i][j] - max_val);
            sum += a[i][j];
        }
        for (int i = 0; i < x; i++) {
            out[i][j] = a[i][j] / sum;
        }
    }
}

void clip_gradients(double** gradients, int x, int y, double clip_value) {
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            if (gradients[i][j] > clip_value) {
                gradients[i][j] = clip_value;
            } else if (gradients[i][j] < -clip_value) {
                gradients[i][j] = -clip_value;
            }
        }
    }
}
void NeuralNets(double** x,int* y,int it)
{
	double** w1 = generate2d(128,500,500.0,128.0);double** b1 = generate2d(128,1,1.0,128.0);
	double** w2 = generate2d(7,128,128.0,7.0);double** b2 = generate2d(7,1,1.0,7.0);
	double** error = allocateMem(7,7000);double** delt = allocateMem(7,7000);
	double** z1 = allocateMem(128,7000);double** a1 = allocateMem(128,7000);
	double** z2 = allocateMem(7,7000);double** a2 = allocateMem(7,7000);
	double** toneHot_y = allocateMem(7,7000);double** deriv = allocateMem(7,7000);
	double** dw2 = allocateMem(7,128);double** db2 = allocateMem(7,1);
	double** dw1 = allocateMem(128,500);double** db1 = allocateMem(128,1);
	double** delt1 = allocateMem(128,7000);double** tw2 = allocateMem(128,7);
	double** ta1 = allocateMem(7000,128);double** tx = allocateMem(7000,500);
	int* predictions = (int*)malloc(7000*sizeof(int));double** temp_derivative = allocateMem(128, 7000);
	double mean[7000];double std[7000];
	printf("%.10f\n",x[0][0]);
	for(int i=0;i<7000;i++)
	{
		for(int j=0;j<500;j++)
		{
			if(isnan(x[j][i]))
			{
				x[j][i] = 0.0;
			}
		}
	}
	for(int i=0;i<it;i++)
	{
		// forward propagation
		prop(w1,x,b1,128,7000,500,z1);
		relu(z1,128,7000,a1);
		prop(w2,a1,b2,7,7000,128,z2);
		softmax(z2,7,7000,a2);

		// loss function
		double** oneHot_y = oneHot(y,7000);
		transpose(oneHot_y,7000,7,toneHot_y);
		for(int i=0;i<7000;i++)
		{
			for(int j=0;j<7;j++)
			{
				error[j][i] = a2[j][i] - toneHot_y[j][i];
			}
		}
		for(int i=0;i<7;i++)
		{
			printf("output_pred : %.10f\n",a2[i][0]);
		}
		//backpropagation starts
		reset(7,7000,delt);
		for(int i=0;i<7;i++)
		{
			for(int j=0;j<7000;j++)
			{
				delt[i][j] = error[i][j];
			}
		}
		transpose(a1,128,7000,ta1);
		backprop_weights(delt,ta1,7,128,7000,dw2);
		backprop_bias(delt,7,1,7000,db2);reset(7,7000,error);

		//error in hidden layer
		transpose(w2,7,128,tw2);
		relu_derivative(a1,128,7000,temp_derivative);
		backprop_error(tw2,delt,128,7000,7,temp_derivative,delt1);reset(7,7000,delt);

		transpose(x,500,7000,tx);
		backprop_weights(delt1,tx,128,500,7000,dw1);
		backprop_bias(delt1,128,1,7000,db1);reset(128,7000,delt1);

		clip_gradients(dw1, 128, 500, 5.0);
		clip_gradients(dw2, 7, 128, 5.0);

		printf("Before update: %.10f\n", w1[0][0]);
		update(w1, dw1, 128, 500);
		printf("After update: %.10f\n", w1[0][0]);
		update(b1,db1,128,1);
		update(w2,dw2,7,128);
		update(b2,db2,7,1);
		get_predictions(a2,7,7000,predictions);
		printf("Iteration : %d\n",i);
		for(int i=0;i<10;i++)
		{
			printf("%d----%d\n",predictions[i],y[i]);
		}
		double accur = accuracy(predictions,y);
		printf("%f\n",accur);

		reset(7, 7000, delt);reset(7, 128, dw2);
		reset(128, 500, dw1);reset(128,7000,delt1);

	}
}

