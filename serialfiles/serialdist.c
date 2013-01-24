#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>

#define MAX_GENES 8000

bool isValidName(char* name);

void trim(char* str);

void usage(void); 

void createGeneListFromFile(FILE* file);

void calculateDistances(char* gene);

int compareDistanceTuple (const void * a, const void * b);

typedef struct {
    double e1[6], e2[6], e3[6];
    char name[20];
} Gene;

typedef struct {
    short geneIndex;
    double distance;
} DistanceTuple;

Gene **geneList;
int geneCount = 0;

int main(int argc, char** argv) {

    if(argc != 2) {
        usage();
        return 1;
    }

    char* selectedGene = "DLL1";

    char* filePath = argv[1];

    FILE *file;

    if(!(file = fopen(filePath, "r"))) {
        //File does not exist
        printf("Unable to open input file %s\n", filePath);
        return 1;
    } 

    //Allocate memory for the gene list
    geneList = malloc(MAX_GENES * sizeof(Gene*));

    for(int i = 0; i < MAX_GENES; i++) {
        geneList[i] = malloc(MAX_GENES * sizeof(Gene));
    }

    //File exists so continue
    createGeneListFromFile(file);
    //Close input file
    fclose(file);

    calculateDistances(selectedGene);

    printf("Done! Results have been stored in the \"output\" folder.\n");

    return 0;
}

void calculateDistances(char* gene) {

    Gene *curr_gene; 
    bool foundGene = false;

    for(int i = 0; i < geneCount; i++) {
        curr_gene = geneList[i];
        if(strcmp(curr_gene->name,gene) == 0) {
            foundGene = true;
            printf("Found gene %s, now calculating distances.\n", gene);
            break;
        }
    }
    
    if(!foundGene) {
        printf("Warning: unable to find specified gene. Program will exit.\n");
        exit(1);
    }
    

    //DistanceTuple **distanceList = malloc(sizeof(DistanceTuple*) * geneCount);

    //for(int i = 0; i < geneCount; i++) {
    //    distanceList[i] = malloc(sizeof(DistanceTuple) * geneCount);
    //}
    //

    DistanceTuple distanceList[geneCount];

    double dist1 = 0.0, dist2 = 0.0, dist3 = 0.0, avgdist = 0.0, sumdist = 0.0;
   // for(int i = 0; i < geneCount; i++) {
        
   //     curr_gene = geneList[i];
        for(int j = 0; j < geneCount; j++) {
            
            Gene *tmp = geneList[j];
            
            dist1 = sqrt(pow(tmp->e1[0] - curr_gene->e1[0],2) + pow(tmp->e1[1] - curr_gene->e1[1],2) +
                         pow(tmp->e1[2] - curr_gene->e1[2],2) + pow(tmp->e1[3] - curr_gene->e1[3],2) +
                         pow(tmp->e1[4] - curr_gene->e1[4],2) + pow(tmp->e1[5] - curr_gene->e1[5],2));

            dist2 = sqrt(pow(tmp->e2[0] - curr_gene->e2[0],2) + pow(tmp->e2[1] - curr_gene->e2[1],2) +
                         pow(tmp->e2[2] - curr_gene->e2[2],2) + pow(tmp->e2[3] - curr_gene->e2[3],2) +
                         pow(tmp->e2[4] - curr_gene->e2[4],2) + pow(tmp->e2[5] - curr_gene->e2[5],2));

            dist3 = sqrt(pow(tmp->e3[0] - curr_gene->e3[0],2) + pow(tmp->e3[1] - curr_gene->e3[1],2) +
                         pow(tmp->e3[2] - curr_gene->e3[2],2) + pow(tmp->e3[3] - curr_gene->e3[3],2) +
                         pow(tmp->e3[4] - curr_gene->e3[4],2) + pow(tmp->e3[5] - curr_gene->e3[5],2));

            avgdist = (dist1 + dist2 + dist3)/3;

            sumdist = dist1 + dist2 + dist3;

            distanceList[j].geneIndex = j;
            //distanceList[j].distance = avgdist;
            distanceList[j].distance = sumdist;
        }

        qsort (distanceList, geneCount, sizeof(DistanceTuple), compareDistanceTuple);

        char fname[40];
        strcpy(fname, "output/");
        strcat(fname, curr_gene->name);
        strcat(fname, ".txt");

        //Write results to file
        FILE * outfile = fopen(fname, "w");
  
        if(!outfile) {
            printf("Warning: File could not be created. Exiting Program.\n");
            exit(1);
        }

        for(int j = 1; j < geneCount; j++) {
            fprintf(outfile, "%d: %s %f\n", j, geneList[distanceList[j].geneIndex]->name, distanceList[j].distance);
        }

        fclose(outfile);
  
    // }
}

void trim(char *str) {
    char *p1 = str, *p2 = str;
    do
        while (*p2 == ' ' || *p2 == '/')
            p2++;
        while (*p1++ = *p2++);
}

int compareDistanceTuple (const void * a, const void * b) {
    
    DistanceTuple *ia = (DistanceTuple*)a;
    DistanceTuple *ib = (DistanceTuple*)b;

    double diff = ia->distance - ib->distance;

    if(diff < 0) {
        return -1;
    } else if (diff == 0.0f) {
        return 0;
    } else {
        return 1;
    }
}

void usage(void) {
    printf("Please enter the path of the input file and no other command line arguments.\n");
}

bool isValidName(char* name) {
   
    if(strcmp(name, "analyte11") == 0) {
        return false;
    } else if (strcmp(name, "siControl") == 0) {
        return false;
    } else if (strcmp(name, "Gene Symbol") == 0) {
        return false;
    } else if (strcmp(name, "siCONT") == 0) {
        return false;
    }else if (*name == 13) {
        return false;
    }

    return true;
}

void createGeneListFromFile(FILE* file) {

    char line[256];


    while(fgets( line, sizeof(line),  file ) != NULL)
    {
        char* tok;
        int len = strlen(line);
        if(len > 0 && line[len-1]=='\n'){ 
            line[len-1] = '\0';
        }
        if(len > 2 && line[len-2]=='\r') {
            line[len-2] = '\0';
        }
        tok = strtok(line,",");
        
        if(tok && isValidName(tok)) {
            
            Gene *currGene = geneList[geneCount];

            strcpy(currGene->name, tok);

            //Eliminate unwanted characters (spaces, slashes)
            trim(currGene->name);

            //Populate experiment 1 array for the gene
            for(int i = 0; i < 6; i++) {
                tok = strtok(NULL,",");
                
                //When there is no data for this experiment, skip to the next experiment
                if(strcmp(tok,"=") == 0) {
                    currGene->e1[i] = 0.0f;
                } else {
                    currGene->e1[i] = atof(tok);
                }
            }

            //Populate experiment 2 array for the gene
            for(int i = 0; i < 6; i++) {
                tok = strtok(NULL,",");

                //When there is no data for this experiment, skip to the next experiment
                if(strcmp(tok,"=") == 0) {
                    currGene->e2[i] = 0.0f;
                } else {
                    currGene->e2[i] = atof(tok);
                }
            }

            //Populate experiment 2 array for the gene
            for(int i = 0; i < 6; i++) {
                tok = strtok(NULL,",");

                //When there is no data for this experiment, skip to the next experiment
                if(strcmp(tok,"=") == 0) {
                    currGene->e3[i] = 0.0f;
                } else {
                    currGene->e3[i] = atof(tok); 
                }
            }

            geneCount++;
        }
    }
}
