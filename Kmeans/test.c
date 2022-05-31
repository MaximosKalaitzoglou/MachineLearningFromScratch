#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#define _XOPEN_SOURCE 600
int x = 0;

void hello(int *y){
    
    if(fork() > 5000){
        printf("Hello from parent !!\n");
    }else{
        printf("Hello from child!!");
    }
}

void main(){
    int y = 5;
    if(fork() != 0){
        y++;
        hello(&y);
    }else{
        x++;
    }
}