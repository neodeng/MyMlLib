#include "SPscala.h"
#include<stdio.h>
#include<string.h>
#include<stdlib.h>

#include "1.h"


struct L
{
	int sn, en;
	double capa, fft;

	double ltt;
	double lf, af, lfn;
};

char** calculate(int origin, int ls, int ns, char** input){
	int link_i,node_i,i;

//	struct L l[Ls];
    struct L *l = (struct L*)malloc( ls*sizeof(struct L) );
//	int pred[Ns + 1];
    int *pred = (int*)malloc( (ns+1)*sizeof(int) );
//	double weight[Ns + 1] = { 0 };
    double *weight = (double*)malloc( (ns+1)*sizeof(double) );
//	int seq[Ns * 2] = { 0 };
    int *seq = (int*)malloc( 2*ns*sizeof(int) );
//	int seq2[Ns + 1] = { 0 };
    int *seq2 = (int*)malloc( (ns+1)*sizeof(int) );
    char** output = (char **) malloc(ns*sizeof(char *));

	for (link_i = 0; link_i<ls; link_i++){
		sscanf(inputc[link_i], "%d,%d,%lf,%lf\n", &l[link_i].sn, &l[link_i].en, &l[link_i].capa, &l[link_i].fft);
	}
	for (link_i = 0; link_i<ls; link_i++){
		l[link_i].ltt=l[link_i].fft;
	}

	for (i = 0; i<ns + 1; i++)
	{
		weight[i] = 9999999999;
		pred[i] = 0;
	}
	weight[origin] = 0; //printf("%lf\n", weight[origin]);
	seq[0] = origin; //printf("%d\n", seq[0]);
	node_i = 0;

    int step=0;
	while (seq[0] != 0){
        step++;
		for (link_i = 0; link_i<ls; link_i++){
			if (l[link_i].sn == seq[0]){
				if ((weight[l[link_i].sn] + l[link_i].ltt)<weight[l[link_i].en]){
					weight[l[link_i].en] = (weight[l[link_i].sn] + l[link_i].ltt);//printf("%d,%lf\n", l[link_i].en, weight[l[link_i].en]);
					pred[l[link_i].en] = l[link_i].sn; //printf("%d,%d\n", l[link_i].sn, l[link_i].en);
					if (exist_or_not(seq, l[link_i].en, ns) == 1){
						if (ever_been_exist_or_not(l[link_i].en, seq2, ns) == 0){
							insert_to_head(seq, l[link_i].en, ns);
						}
						else{
							insert_to_end(seq, l[link_i].en);
						}
					}
				}
			}
		}
		seq2[seq[0]] = seq[0]; //printf("%d,%d\n", seq[0], seq2[seq[0]]);
		for (i = 0; i < ns; i++){
			seq[i] = seq[i + 1];
		}
	}

	for (i = 1; i < ns+1; i++){
		sprintf(output[i-1],"%d,%d\n", i, pred[i]);
	}

    free(l);
    free(pred);
    free(weight);
    free(seq);
    free(seq2);

	return output;
}

/*===================== From Scala Call Shorted function ==========================*/
///*
JNIEXPORT jobjectArray JNICALL Java_SPscala_SPMain
(JNIEnv* env, jobject obj, jint origin, jint ls, jint ns, jobjectArray input) {
    char **inputc = (char **) malloc(ls*sizeof(char *));
    jsize j = 0;
    for (j=0; j<len; j++) {
        jstr = (*env)->GetObjectArrayElement(env, input, j);
        inputc[j] = (char *)(*env)->GetStringUTFChars(env, jstr, 0);
    }
    char** outputc = calculate(origin, ls, ns, inputc);
    jclass stringClass = (*env)->FindClass(env, "java/lang/String");
    jobjectArray output = (*env)->NewObjectArray(env, ns, stringClass, 0);
    for (j=0; j<ns; j++) {
        (*env)->SetObjectArrayElement(env, output, j, (*env)->NewStringUTF(env, outputc[j]));
    }
    free(inputc);
    free(outputc);
    return output;
}
//*/

