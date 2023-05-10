#include<stdio.h>
#include<time.h>
#include<stdlib.h>

int main() {
  usleep(100 * 1000);
  printf("Sleep done!");
  if(getchar() == 'c' && getchar() == 'a' && getchar() == 'b' ){
    abort();
  }
}
