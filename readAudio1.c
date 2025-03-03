#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include "neuralNets.h"
char* path(char* directory,char* name)

{
    int newLength = strlen(directory) + strlen(name) + 2; // +1 for the slash
    char *newPath = (char *)malloc(newLength);
    // Build the new directory path
    snprintf(newPath, newLength, "%s/%s", directory, name);
    // Print the new directory path
    return newPath;
}
int main() {
    double** xt = (double**)malloc(501*sizeof(double*));
    for(int i=0;i<501;i++)
	xt[i] = (double*)malloc(7000*sizeof(double));
    int* y = (int*)malloc(7000*sizeof(int));
    char* dir = "/home/saba/speech_data/output";
    DIR* direc = opendir(dir);
    struct dirent *entry;
    FILE *file;
    
    char line[1000];
    int size = 500*204;
    double val[size];int c = 0;int l=0;
    while((entry = readdir(direc)) != NULL){
	if(strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0){
		continue;
	}
	printf("%d\n",c);
	char* data = path(dir,entry->d_name);
    	file = fopen(data,"r");

    	int count = 0;int f=0;
    	while(fgets(line,sizeof(line),file)){
		char* ptr = line;
		double value;
		while(sscanf(ptr,"%lf",&value) == 1){
			if(count==size)
			{break;f=1;}
			val[count++] = value;
			while(*ptr != '\t' && *ptr != '\0'){
				ptr++;
			}
			if(*ptr == '\t') ptr++;
		}
		if(f==1)
		{break;}
   	}
       fclose(file);
       double mfc[500];int mcount = 0;
       for(int i=0;i<count;i=i+204)
       {
	   double sum = 0;
	   for(int j=i;j<i+204;j++)
	   {
		sum += val[j];
	   }
	   mfc[mcount] = sum/204;mcount=mcount+1;
      }
      for(int i=0;i<500;i++)
      {
	   if(isnan(mfc[i]))
	   {
		printf("%d---%d\n",c,i);
		xt[i][c] = 0.0;
	   }
     	   xt[i][c] = mfc[i];
      }
      if((c+1) % 1000 == 0)
	{xt[500][c] = l;l=l+1;}
      else
        {xt[500][c] = l;}
      c=c+1;
    }
    srand(time(NULL));
    for(int i=7000-1;i>0;i--)
    {
	int j = rand() % (i+1);
	for(int k=0;k<501;k++)
	{
		double temp = xt[k][i];
		xt[k][i] = xt[k][j];
		xt[k][j] = temp;
	}
    }
    double** xt_train = (double**)malloc(500*sizeof(double*));
    for(int i=0;i<500;i++)
	xt_train[i] = (double*)malloc(7000*sizeof(double));
    for(int i=0;i<500;i++)
    {
	for(int j=0;j<7000;j++)
	{
		xt_train[i][j] = xt[i][j];
	}
    }
    for(int i=0;i<7000;i++)
    { y[i] = xt[500][i];}

    for(int i=0;i<501;i++)
    { free(xt[i]);}
    free(xt);
    NeuralNets(xt_train,y,100);
    return 0;
}
