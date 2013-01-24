#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#define MAX_GENES 25000
#define NUMBER_OF_RESULTS 10

bool isValidName(char* name);

void trim(char* str);

void usage(void); 

void createGeneListFromFile(FILE* file);

void calculateDistances(char* gene);

void sortAndPrint();

int compareDistanceTuple (const void * a, const void * b);

typedef struct {
    double e1[6], e2[6], e3[6];
    char name[20];
} Gene;

typedef struct {
    unsigned int geneOne;
    unsigned int geneTwo;
    double distance;
} DistanceTuple;

DistanceTuple** distanceMatrix;

Gene **geneList;
int geneCount = 0;

int main(int argc, char** argv) {

    if(argc != 2) {
        usage();
        return 1;
    }

    char* selectedGene = "DYRK1A";

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

    printf("This is genecount: %d\n", geneCount);

    //Allocate memory for the distance matrix
    distanceMatrix = malloc(geneCount * sizeof(DistanceTuple*));

    for(int i = geneCount - 1; i >= 0; i--) {
        distanceMatrix[geneCount - i - 1] = malloc(i * sizeof(DistanceTuple));
    }

    calculateDistances(selectedGene);

    sortAndPrint();

    printf("Done! Results have been stored in the \"output\" folder.\n");

    return 0;
}

void sortAndPrint() {
    printf("In sort now\n");
    //Need to reconstruct a templist from the distance matrix for each gene
    DistanceTuple* tempDistanceList = malloc((geneCount - 1) * sizeof(DistanceTuple));
    int distanceIndex;

    for(int i = 0; i < geneCount; i++) {

        Gene* curr_gene = geneList[i];
        distanceIndex = 0;
        int row, col;

        row = i - 1;
        col = 0;

        //weird diagonal stuff
        while(row >= 0 && col <= i - 1) {

            tempDistanceList[distanceIndex] = distanceMatrix[row][col];
            distanceIndex++;
            
            row--;
            col++;
        }

        //rest of the column
        for(int j = 0; j < geneCount - i - 1; j++) {

            tempDistanceList[distanceIndex] = distanceMatrix[i][j];
            tempDistanceList[distanceIndex].geneOne = distanceMatrix[i][j].geneTwo;
            distanceIndex++;
        }

        qsort (tempDistanceList, geneCount - 1, sizeof(DistanceTuple), compareDistanceTuple);

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

        for(int j = 0; j < NUMBER_OF_RESULTS; j++) {
            fprintf(outfile, "%d: %s %f\n", j+1, geneList[tempDistanceList[j].geneOne]->name, tempDistanceList[j].distance);
        } 

        fclose(outfile);
      
    }
}

void calculateDistances(char* gene) {

    Gene *curr_gene; 

    double dist1 = 0.0, dist2 = 0.0, dist3 = 0.0, sumdist = 0.0;
    
    for(int i = 0; i < geneCount; i++) {
        
        curr_gene = geneList[i];

        int distColIndex = 0;

        for(int j = i+1; j < geneCount; j++) {
            
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

            sumdist = dist1 + dist2 + dist3;

            distanceMatrix[i][distColIndex].geneOne = i;
            distanceMatrix[i][distColIndex].geneTwo = j;
            distanceMatrix[i][distColIndex].distance = sumdist;

            distColIndex++;
        }
    }
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
