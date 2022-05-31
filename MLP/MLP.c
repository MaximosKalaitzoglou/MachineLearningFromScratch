#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>



#define D 2 //input neurons 

#define OUT 4
#define MAX 4000
#define max(x, y) (x > y ? x : y) //max function

#define epochs 5000


//globally defined values 
float celling; // minimum difference between errors in t+1 , t epochs
int H1,H2,H3; // number of neurons in hidden layers
int HiddenLayers; // number of hidden layers
int num_of_layers; // number of layers
float learningRate; // learningRate
float previous_error = 0; // previous error meaning error of the previous epoch of training
int Mode = -1; // mode is used to know if training will be serial , group training  or mini_batch training
int *neuron_in_layers; // array of neurons in each layer
int B; //denominator of MAX values
int Batches; // MAX/B 

//inputs for train_datasets and test_datasets 
typedef struct
{
    float *x1; //input x1,x2 
    int t; //corresponding class
} inputs;

//neuron struct for each neuron -> weight ->output,bias,input
typedef struct{
    
    float out;  
    float in;
    float delta_err;
    
    
}neuron;


//layer struct contains number of neurons in each layer and a pointer to neuron struct

typedef struct Lay{
    int numOfneurons;
    neuron *n;
    float *bias;
    float *deltaBias;
    float *weight;
    float *deltaWeight;
    int activation;
}layers;



inputs *data_train;
inputs *data_test;
layers *layer;

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



float my_tanh(float in){
    float output;
    output = (exp(in) - exp(-in))/(exp(in) + exp(-in));
    //printf("output = %f\n",output);
    //exit(0);
    return output;
}

float tanh_prime(float in){
    float out = 1 - square(my_tanh(in));
    return out;
}


//relu activation function 
float relu(float in){
    float out;
    
    out = max(0.0,in);
   
    return out;
}


float sigmoid(float in){

    return (1.0 / (1.0 + exp(-in)));
}

float sigmoid_prime(float output){
    return sigmoid(output) *(1-sigmoid(output));
}

float relu_prime(float output){
    if(output > 0){
        return 1;
    }
    return 0;
}




void print_data(inputs *p){
  
    for (int i = 0; i < MAX; i++)
    {
        printf("%d : (x1,x2) = (%f,%f)\t t = %d\n", (i + 1),p[i].x1[0], p[i].x1[1],p[i].t);
    }
}

float my_abs(float x, float y)
{
    if (x > y)
    {
        return x - y;
    }
    return y - x;
}


float calculate_total_Error(float *target,struct Lay x){
    float *Error;
    Error = malloc(OUT*sizeof(float));
    float total_Error = 0.0;
    for(int i = 0;i<OUT;i++){
        //printf("target %d = %f\n",i,target[i]);
        //printf("out %d = %f\n",i,out[i]);
        Error[i] = 0.25*square(target[i] - x.n[i].out);
        //printf("Total error for output %d = %f\n",i,Error[i]);
        total_Error += Error[i];
    }
    
    return total_Error;
}

float *set_target(inputs data){
    int target = data.t - 1;
   // printf("target = %d\n",target);
    float *t = malloc(OUT*sizeof(float));
    for(int i = 0;i<OUT;i++){
        t[i] = 0.0;
    }
    t[target] = 1.0;
    
    return t;
}


void get_output(inputs data_test,layers output,int k){
   
    printf("Class expected %d\n",data_test.t);
    float c = 0.0;
    int t = -1;
    for (int i = 0; i < k; i++)
    {
        if (c < output.n[i].out)
        {
            
            c = output.n[i].out;
            t = i;
        }
    }
    printf("class computed by AI = %d\n", t + 1);
    for (int i = 0; i < OUT; i++)
    {
        printf("output x%d = %f\n", i,output.n[i].out);
    }
}


float trans_derivative(float output){
    float result = output*(1-output);
    return result;
}

//if B == MAX then we have group update of weights equal to B 
//if B == 1 then we have serial update of weights
//else we have mini_batch update of weights

void set_Mode(){
    if(Batches == MAX){
        Mode = 0;
    }else if (Batches == 1){
        Mode = 1;
    }else{
        Mode = 2;
    }
}


int act_hidden = 0;
int act_out = 0;


bool not_in_array(int value,int *tmp){
    for(int i = 0;i < 3;i++){
        if(value == tmp[i]){
            return false;
        }
    }
    return true;
}

void activationFunction(){
    int n = 0;
    int x = 0;
    int tmp[3] = {1,2,3};
    while(not_in_array(n,tmp) || not_in_array(x,tmp)){
        printf("Please choose which of the following activation functions you want for the HiddenLayers and Output layers respectively:\n");
        printf("Sigmoid : 1\nTanh:2\nRelu:3\n");
        scanf("%d",&n);
        scanf("%d",&x);
    }
    act_hidden = n;
    act_out = x;
    
}


void update_weights(int h){
    //printf("Layer for update %d\n",h);
    int weight_count = 0;
    if(Batches == 1){
        for(int i = 0;i < layer[h].numOfneurons;i++){
            for(int j = 0;j < layer[h-1].numOfneurons;j++){
                float alpha = learningRate*layer[h].n[i].delta_err*layer[h-1].n[j].out;
            
            //printf("Alpha = %f\n",alpha);
                layer[h].weight[weight_count] -= alpha;
                weight_count++;
            }
        //layer[h].deltaBias = layer[h].n[i].delta_err;
        //layer[h].bias[i] -= layer[h].deltaBias;
        }
    }else{
        for(int i = 0;i<layer[h].numOfneurons;i++){
            for(int j = 0;j < layer[h-1].numOfneurons;j++){
                float alpha = learningRate*layer[h].n[i].delta_err*layer[h-1].n[j].out;
                layer[h].deltaWeight[weight_count] += alpha;
                weight_count++;
            }
            //layer[h].deltaBias[i] += learningRate*layer[h].n[i].delta_err;
        }
    }
    
}



float get_activation(layers y,float in,bool prime){
    float output;
    if(prime){
        if(y.activation == 1){
            output = sigmoid_prime(in);
        }else if (y.activation == 2){
            output = tanh_prime(in);
        }else{
            output = relu_prime(in);
        }
        return output;
    }else{
        if(y.activation == 1){
            output = sigmoid(in);

        }else if (y.activation == 2){
            output =  my_tanh(in);
        }else{
            output = relu(in);
        }
        
        return output;
    }
    
}

struct Lay forward_pass(float *x,int d, float *y,int k)
{
    //initialize starting layer input and output
    layer[0].n[0].in = x[0];
    layer[0].n[1].in = x[1];
    layer[0].n[0].out = layer[0].n[0].in;
    layer[0].n[1].out = layer[0].n[1].in;


    int i,j,h;
    //feed forward 
    for(h = 1;h < num_of_layers;h++){
       // printf("LAYER %d\n",h);
        float sum = 0;
        int weight_count = 0;
        for(i = 0;i < layer[h].numOfneurons;i++){
            for(j = 0;j < layer[h-1].numOfneurons;j++){
                sum += layer[h].weight[weight_count]*layer[h-1].n[j].out;
                weight_count++;
            }
            layer[h].n[i].in = sum + layer[h].bias[i];
            layer[h].n[i].out = get_activation(layer[h],layer[h].n[i].in,false);
            //printf("output %d = %f\n",i,layer[h].n[i].out);
        }
    }
    //get_output(data_test[0],layer[3],4);
    return layer[num_of_layers-1];
}



void backprop(float *x,int d,float *t ,int k){
    //output layer
    //printf("OUTPUT LAYER\n");
    int out = num_of_layers-1;
    for(int i = 0;i < layer[out].numOfneurons;i++){
        //linear activation 
        layer[out].n[i].delta_err = get_activation(layer[out],layer[out].n[i].in,true)*(layer[out].n[i].out - t[i]);
       // printf("delta error %d = %f\n",i,layer[3].n[i].delta_err);
    }
    update_weights(out);

    //hidden layers
    for(int h = num_of_layers-2;h>0;h--){
        //printf("LAYER %d\n",h);
        float sum = 0;
        int weight_count = 0;
        for(int i = 0;i < layer[h].numOfneurons;i++){
            for(int j = 0;j < layer[h+1].numOfneurons;j++){
                sum += layer[h+1].weight[weight_count]*layer[h+1].n[j].delta_err;
                weight_count++;
            }
            layer[h].n[i].delta_err = sum * get_activation(layer[h],layer[h].n[i].in,true);
            //printf("Delta error %d = %f\n",i,layer[h].n[i].delta_err);
        }
        update_weights(h);
        
    }
    
}








void train_model(){
    
   for(int e = 0;e < epochs;e++){
       float error = 0;
       layers y;
       float *t = malloc(OUT*sizeof(float));
       for(int i = 0;i < MAX;i++){
           y = forward_pass(data_train[i].x1,D,t,OUT);
           t = set_target(data_train[i]);
           error += calculate_total_Error(t,y);
           backprop(data_train[i].x1,D,t,OUT);
       }
       
       
       if(Batches == MAX){
           for(int h = 1;h < num_of_layers;h++){
               int weight_count = 0;
               for(int i = 0;i < layer[h].numOfneurons;i++){
                   for(int j = 0;j < layer[h-1].numOfneurons;j++){
                       layer[h].weight[weight_count] -= layer[h].deltaWeight[weight_count];
                       layer[h].deltaWeight[weight_count] = 0;
                       weight_count++;
                   }
               }
           }
       }else{
           for(int h = 1;h < num_of_layers-1;h++){
               for(int i = 0;i < layer[h].numOfneurons;i++){
                   layer[h].n[i].delta_err = 0;
               }
           }
       }
       printf("Epoch = %d , learning Rate = %f, error = %f\n",e,learningRate,error);
       if(my_abs(error,previous_error) < celling && e >= 700){
            printf("Celling reached!!\n");
            break;
        }
        previous_error = error;

   }
  
    
}




void mini_batch(){
    
    layers y;
    float *t = malloc(OUT*sizeof(float));
    printf("batch iterations = %d\n",Batches);
    for(int e = 0;e < epochs;e++){
        float error = 0;
         for(int i = 0;i <B;i++){
            
            
            for(int j = Batches*i; j < Batches*(i+1);j++){
                y = forward_pass(data_train[j].x1,D,t,OUT);
                t = set_target(data_train[j]);
                error += calculate_total_Error(t,y);
                backprop(data_train[j].x1,D,t,OUT);
            }
            for(int h = 1;h < num_of_layers;h++){
                int weight_count = 0;
                for(int o = 0; o < layer[h].numOfneurons;o++){
                    for(int l = 0;l < layer[h-1].numOfneurons;l++){
                        layer[h].weight[weight_count] -= layer[h].deltaWeight[weight_count];
                        //printf("layer %d weight %d = %f\n",h,weight_count,layer[h].weight[weight_count]);
                        layer[h].deltaWeight[weight_count] = 0;
                        weight_count++;
                    }
                   // layer[h].bias[o] -= layer[h].deltaBias[o];
                }
            }

        }
        printf("Epoch %d,learning Rate %f, error %f\n",e,learningRate,error);
        if(my_abs(error,previous_error) < celling && e >= 700){
            printf("Celling reached!!\n");
            break;
        }
        previous_error = error;
    }
}



void Init_layers(){
    int layer_num = -1;
    while(layer_num != 2 && layer_num != 3){
        printf("Please input (2 or 3) for 2 hidden layers or 3 hidden layers\n");
        scanf("%d",&layer_num);
    }
    if(layer_num == 2){
        int n1,n2;
        HiddenLayers = 2;
        printf("Give number of neurons for layer H1:\n");
        scanf("%d",&n1);
        H1 = n1;
        printf("Give number of neurons for layer H2:\n");
        scanf("%d",&n2);
        H2 = n2;
        
    }else{
        int n1,n2,n3;
        HiddenLayers = 3;
        printf("Give number of neurons for layer H1:\n");
        scanf("%d",&n1);
        H1 = n1;
        printf("Give number of neurons for layer H2:\n");
        scanf("%d",&n2);
        H2 = n2;
        printf("Give number of neurons for layer H3:\n");
        scanf("%d",&n3);
        H3 = n3;
        
        
    }
    num_of_layers = HiddenLayers + 2;
    neuron_in_layers = malloc(num_of_layers*sizeof(int));
    
    if(HiddenLayers == 2){
        int tmp[4] = {D,H1,H2,OUT};
        for(int i = 0;i < num_of_layers;i++){
            neuron_in_layers[i] = tmp[i];
        }
    }else{
        int tmp[5] = {D,H1,H2,H3,OUT};
        for(int i = 0;i < num_of_layers;i++){
            neuron_in_layers[i] = tmp[i];
        }
    }
    
    
    
}


void init_structure(){
    layer = malloc(num_of_layers * sizeof(layers));
    for(int i = 0;i < num_of_layers;i++){
        layer[i].n = malloc(neuron_in_layers[i]*sizeof(neuron));
        if(i != 0){
            layer[i].weight = malloc(neuron_in_layers[i]*neuron_in_layers[i-1]*sizeof(float));
            layer[i].deltaWeight = malloc(neuron_in_layers[i]*neuron_in_layers[i-1]*sizeof(float));
            if(i != num_of_layers-1){
                layer[i].activation = act_hidden;
               
            }else{
                layer[i].activation = act_out;
            }
        }
        //neuron malloc
        layer[i].bias = malloc(neuron_in_layers[i]*sizeof(float));
        for(int k = 0;k < neuron_in_layers[i];k++){
            layer[i].bias[k] = 1.0;
        }
        layer[i].numOfneurons = neuron_in_layers[i];
        layer[i].deltaBias = malloc(neuron_in_layers[i]*sizeof(float));
        for(int j = 0;j < neuron_in_layers[i];j++){
            layer[i].n[j].delta_err = 0;
            layer[i].n[j].in = 0;
            layer[i].n[j].out = 0;
        }

    }

    
}



void initialize_data(){

    data_train = malloc(4000 * sizeof(inputs));

    data_test = malloc(4000 * sizeof(inputs));

    for (int i = 0; i < MAX; i++)
    {

        data_train[i].x1 = malloc(D * sizeof(float));

        data_test[i].x1 = malloc(D * sizeof(float));
    }

    FILE *f = fopen("TrainSet.txt", "r");

    if (f == NULL)

    {
        exit(printf("file not found\n"));
    }
    int index = 0;

    while (index < MAX)
    {
        if (fscanf(f, "%f %f %d", &data_train[index].x1[0], &data_train[index].x1[1], &data_train[index].t) == 3)
        {
            index++;
        }
    }
    fclose(f);

    //activationFunc(p.hidden_activation_functions[0],input,output);

    // printf("data_train set : ");
    // print_data(data_train);

    FILE *test = fopen("TestSet.txt", "r");

    if (test == NULL)
    {
        exit(printf("file not found\n"));
    }
    index = 0;

    while (index < MAX)
    {
        if (fscanf(test, "%f %f %d", &data_test[index].x1[0], &data_test[index].x1[1], &data_test[index].t) == 3)
        {
            index++;
        }
    }
    fclose(test);

    
}



void initialize_weights(){
    for(int h = 1;h<num_of_layers;h++){
        int n = 0;
        for(int i = 0;i < layer[h].numOfneurons;i++){
            for(int j = 0;j<layer[h-1].numOfneurons;j++){
                float w = randfrom(-1.0,1.0);
                layer[h].weight[n] = w;
                n++;
                
        
                //printf("i = %d , j = %d\n",i,j);
            }
            //layer[h].weight[i] = 1;
        
        }
        

    }    
    
}



void get_accuracy(float ac,int Mode){
    FILE *acy = fopen("Accuracy.txt","a");
    if(HiddenLayers == 2){
        fprintf(acy,"Model with %d hidden layers and with,number of neurons in each hidden layer: H1 = %d , H2 = %d\n",HiddenLayers,layer[1].numOfneurons,layer[2].numOfneurons);
    }else{
        fprintf(acy,"Model with %d hidden layers and with,number of neurons in each hidden layer: H1 = %d , H2 = %d , H3 = %d\n",HiddenLayers,layer[1].numOfneurons,layer[2].numOfneurons,layer[3].numOfneurons);
    }
    
    if(Mode != 1 && Mode != 0){
        fprintf(acy,"Model Batch size used = MAX/%d\n",B);
    }else if (Mode == 1){
        fprintf(acy,"Model is training with B = 1 so we have a serial update of weights\n");
    }else{
        fprintf(acy,"Model is training with B = MAX so we have a group update of weights\n");
    }
    
    fprintf(acy,"Model total Accuracy = %f%%\n",ac*100);
    fprintf(acy,"Learning Rate used = %f\n",learningRate);
    fprintf(acy,"----------------------------------\n");
    fclose(acy);
}


int correct_values = 0;
int wrong_values = 0;

void training_results(int size){
    FILE *f = fopen("correct.txt","w");
    FILE *g = fopen("false.txt","w");
    float accuracy = 0.0;
    int correct = 0;
    int false_n = 0;
    int C1 = 0;
    int C2 = 0;
    int C3 = 0;
    int C4 = 0;
    for(int i = 0;i<size;i++){
        layers y;
        y = forward_pass(data_test[i].x1,D,NULL,OUT);
        //printf("Class expected %d , %d\n",i, data_test[i].t);
        float c = 0.0;
        int t = -1;
        if(data_test[i].t == 1){
            C1 ++;
        }else if(data_test[i].t == 2){
            C2++;
        }else if(data_test[i].t == 3){
            C3++;
        }else{
            C4++;
        }
        for (int i = 0; i < 4; i++)
        {
            if (c < y.n[i].out)
            {
                
                c = y.n[i].out;
                t = i;
            }
        }
        //printf("class computed by AI = %d\n", t + 1);
        if(t+1 == data_test[i].t){
            correct += 1;
            fprintf(f,"%f %f +\n",data_test[i].x1[0],data_test[i].x1[1]);
        }else{
            false_n += 1;
             fprintf(g,"%f %f -\n",data_test[i].x1[0],data_test[i].x1[1]);
        }
    }
    printf("found true for %d values given : %d\n",size, correct);
    printf("found false for %d values given : %d\n",size, false_n);
    float ac = (float) correct / (float) MAX;
    printf("Accuracy for %d values = %f or %g%%\n ",size, ac,ac*100);
    printf("DATA BASE HAS A TOTAL OF \n%d type 1 sets\n %d type 2 sets\n %d type 3 sets\n %d type 4 sets\n",C1,C2,C3,C4);
    layers y;
    get_accuracy(ac,Mode);
    fclose(f);
    fclose(g);
    correct_values = correct;
    wrong_values = false_n;
    //y = forward_pass(data_test[15].x1,D,NULL,OUT);
    //get_output(data_test[15],y.n->out,OUT);
}


void setUpForTraining(){
    printf("Please input desired learning Rate for model Training:\n");
    float l;
    scanf("%f",&l);
    learningRate = l;
    printf("Please input the size of B denominator of the data_training's size:\n");
    printf("Using a B = 1 means we have N/B = N group training , and using B = 4000 means we have N/B = 1 serial training\n");
    scanf("%d",&B);
    int N = MAX;
    Batches = N/B;
    printf("Please input celling value (normaly between 0.01,0.001 etc)\n");
    scanf("%f",&celling);
    
}


void printBestModel(){
    int i,j;
    FILE *gnuplot = fopen("MLP.txt", "w");
    FILE *r = fopen("correct.txt","r");
    FILE *g = fopen("false.txt","r");
    float x0,x1;
    fprintf(gnuplot,"set style line 1 lc rgb 'green' pt 9 ps 1\n");
    fprintf(gnuplot,"set style line 2 lc rgb 'red' pt 29 ps 1\n");
    fprintf(gnuplot, "plot '-' with points ls 1,'-' with points ls 2\n");
    for(int i = 0;i <correct_values;i++){
        if(fscanf(r,"%f %f\n",&x0,&x1) == 2){
            fprintf(gnuplot,"%f %f\n",x0,x1);
        }
        
    }
    fprintf(gnuplot, "e\n");
    for(int j = 0;j < wrong_values;j++){
        if(fscanf(g,"%f %f\n",&x0,&x1) == 2){
            fprintf(gnuplot,"%f %f\n",x0,x1);
        }
        
    }
    
    
    fprintf(gnuplot, "e\n");
    fflush(gnuplot);
    fclose(r);
    fclose(g);

}

int main(){


//read train data and test data from files
    Init_layers();
    activationFunction();
    initialize_data();
    printf("Successfully initialized data for training set !!\n");
    init_structure();
    printf("Successfully initialized structure!!\n");
    layers y;
    // initialize weight matrices
    initialize_weights();
    printf("Succesfully initialized weights !!@!!\n");
    
    float *t = malloc(OUT*sizeof(float));
    
    setUpForTraining();
    set_Mode();
    printf("Ready to Train!!\n");
    
   
    if(B == 1 || B == MAX){
        train_model();
    }else{
        mini_batch();
    }
    
    training_results(MAX);
    
    printBestModel();    
    return 0;
}




