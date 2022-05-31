#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>



#define N 1200
int M = -1;

float previous_error = 10000;



typedef struct Previous{
    float p0;
    float p1;

}previous;

typedef struct Means{
    float m0;
    float m1;
    previous prev;
    
}means;




typedef struct Points{
    float x0;
    float x1;
    int t;
}points;

points *p;
means *m;


float randfrom(float min, float max)
{
    float range = (max - min);
    float div = RAND_MAX / range;
    return min + (rand() / div);
}

float square(float dVal)
{
    return (dVal * dVal);
}



void init_structure(){
    for(int i = 0;i < M;i++){
        int random = randfrom(0,1200);
        m[i].m0 = p[random].x0;
        m[i].m1 = p[random].x1;
        m[i].prev.p0 =  m[i].m0;
        m[i].prev.p1 =  m[i].m1;
        
    }
    
}

void read_data(){
    FILE *f = fopen("meansSet.txt", "r");

    if (f == NULL)

    {
        exit(printf("file not found\n"));
    }
    int index = 0;

    while (index < N)
    {
        if (fscanf(f, "%f %f", &p[index].x0, &p[index].x1) == 2)
        {
            index++;
        }
    }
    fclose(f);
}


int find_min_dist(float *e){
    float min = 100;
    float class = -1;
    for(int i = 0;i<M;i++){
        if(e[i] < min){
            min = e[i];
            class = i;
        }
    }
    return class;
}


void euclidean_distance_from_all_means(int point){
    float *e = malloc(M*sizeof(float));
    for(int i = 0;i < M;i++){
        float d1 = m[i].m1 - p[point].x1;
        float d2 = m[i].m0 - p[point].x0;
        e[i] = square(d1) + square(d2);
    }
    
    int t = find_min_dist(e);
    p[point].t = t;

}

float euclidean_distance_from_class(){
    float sum = 0;
    for(int i = 0;i<N;i++){
        float d1 = m[p[i].t].m1 - p[i].x1;
        float d2 = m[p[i].t].m0 - p[i].x0;
        float e = square(d1) + square(d2);
        sum += e;
        
    }
    return sum;
}

void update_means(){
    for(int j = 0;j<M;j++){
        float mean1 = 0;
        float mean2 = 0;
        float counter = 0;
        for(int i = 0;i < N;i++){
            if(p[i].t == j){
                mean1 += p[i].x1;
                mean2 += p[i].x0;
                counter += 1;
            }
        }
        mean1 /= counter;
        mean2 /= counter;
        m[j].m1 = mean1;
        m[j].m0 = mean2;
    
    }
    
}

void print_classes(){
    FILE *f = fopen("class.txt","w");
    
    
    for(int i = 0;i < N;i++){
        fprintf(f,"class %d\n",p[i].t);
    }
    fclose(f);
}


void print_means(){
    for(int i = 0;i<M;i++){
        printf("Mean %d\n",i);
        printf("m0 = %f\tm1 = %f\n",m[i].m0,m[i].m1);
    }
}


void update_mean_previous(){
    for(int j = 0;j < M;j++){
        m[j].prev.p0 = m[j].m0;
        m[j].prev.p1 = m[j].m1;
    }
}


void Kmeans(){
    
    float total_error = 0;
   
   
    for(int t = 0;t < 1000;t++){
        int break_flag = 0;
        printf("%d\n",t);
        for(int i = 0;i<N;i++){
            euclidean_distance_from_all_means(i);
        }
        update_means();
        for(int j = 0;j < M;j++){
            if(m[j].m0 == m[j].prev.p0 && m[j].m1 == m[j].prev.p1){
                
                break_flag += 1;
            }
        }
        if(break_flag == M){
            printf("celling reached\n");
            break;
        }
        update_mean_previous();
        total_error = euclidean_distance_from_class();
        printf("epoch %d ,total error %f\n",t,total_error);
        
    }
     
    if(total_error <= previous_error){
        print_classes();
        printf("Overwrite current_error = %f , previous = %f\n",total_error,previous_error);
        previous_error = total_error;
    }
    
}




void plot_data(){
    
    int i,j;
    FILE *gnuplot = fopen("gnuplot.txt", "w");
    fprintf(gnuplot,"set style line 1 pt 1 ps 1\n");
    fprintf(gnuplot,"set style line 2 pt 7 ps 1.5\n");
    fprintf(gnuplot, "plot '-' with points palette,'-' with points ls 2\n");
    for (i = 0; i <N; i++)
        fprintf(gnuplot, "%g %g %d\n",p[i].x0,p[i].x1,p[i].t);
    fprintf(gnuplot, "e\n");
    for(j = 0;j < M;j++)
        fprintf(gnuplot, "%g %g %d\n",m[j].m0,m[j].m1,9);
    
    fprintf(gnuplot, "e\n");
    fflush(gnuplot);
}

void print_final_means(){
    for(int i = 0;i < M;i++){
        printf("Mean %d , (x1,x2) = (%f,%f)\n",i,m[i].m0,m[i].m1);
    }
}

void save_error(){
    FILE *e = fopen("errors.txt","a");
    fprintf(e,"Number of groups M = %d , minimum error after 20 iterations = %f\n",M,previous_error);
    fclose(e);
}


void main(){

	printf("Please choose a number of means M:\nCommonly used values are M = {3,5,7,9,11,13}\n");
	scanf("%d",&M);
    p = malloc(N*sizeof(points));
    read_data(p);
    m = malloc(M*sizeof(means));
    
    init_structure();
    float previous = 0;
    for(int i = 0;i < 20;i++){
        printf("Iteration %d for M = %d\n",i,M);
       // print_means();
        init_structure();
        Kmeans();
        
        
    }
    print_final_means();
    //IF you want to plot the data call gnuplot gnuplot.txt 
    plot_data();
    //save_error();
    //Kmeans();
    //print_means(m);
    //print_classes(p);
}