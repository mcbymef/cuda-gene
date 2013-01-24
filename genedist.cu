#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include<cuda.h>
#include "genedist.h"

char** nameList;

int geneCount = 0;

//Number of results that will be saved for each gene
int numberOfResults = 10;

//Determines if the program will run for all genes or a single one
unsigned char calculateAllGenes = 1;

//May be unable to use CUDA due to GPU memory unavilability
unsigned char canUseCuda;

char* selectedGene;

//Texture memory will be used to store gene information
texture<int2, 1, cudaReadModeElementType> geneTex;

__global__ void calculateDistanceGPU(double* distance_d, int geneCount) {


    __shared__ double s_genes[32 * REPLICATES_PER_GENE];
    __shared__ double results[1024];

    double dist = 0.0;

    int geneIndex = blockIdx.x * blockDim.x + threadIdx.x;

    //Fill own gene (these memory accesses are not very much fun but can't be avoided if we use texture memory)
    double curr_gene[REPLICATES_PER_GENE];

    if(geneIndex < geneCount) {

        for(int i = 0; i < REPLICATES_PER_GENE; i++) {
        	int2 v = tex1Dfetch(geneTex, geneIndex * REPLICATES_PER_GENE + i);
        	curr_gene[i] = __hiloint2double(v.y, v.x);
        }

        int top = (geneCount % 32 == 0) ? geneCount/32 : geneCount/32 + 1;

        for(int i = 0; i < top; i++) {

            //Fill the shared input array collaboratively 
        	for(int j = 0; j < REPLICATES_PER_GENE; j++) {

        		//Make sure the gene being loaded is in bounds (the number of genes will likely not be divisible by 32, so the last block
                //will not have 32 valid array indeces to access
        		if(!(i == top - 1 && threadIdx.x > 3 )) {
        		    int2 v = tex1Dfetch(geneTex, (i * 32 * REPLICATES_PER_GENE) + (threadIdx.x * REPLICATES_PER_GENE) + j);
        		    s_genes[threadIdx.x * REPLICATES_PER_GENE + j] =  __hiloint2double(v.y, v.x);
        		}
        	}

            for(int j = 0; j < 32; j++) {

            	int offset = REPLICATES_PER_GENE * j;

            	dist	=    __dsqrt_rz(pow(s_genes[0 + offset] - curr_gene[0],2) + pow(s_genes[1 + offset] - curr_gene[1],2) +
            			     pow(s_genes[2 + offset] - curr_gene[2],2) + pow(s_genes[3 + offset] - curr_gene[3],2) +
                             pow(s_genes[4 + offset] - curr_gene[4],2) + pow(s_genes[5 + offset] - curr_gene[5],2)) +
                             __dsqrt_rz(pow(s_genes[6 + offset] - curr_gene[6],2) + pow(s_genes[7 + offset] - curr_gene[7],2) +
                             pow(s_genes[8 + offset] - curr_gene[8],2) + pow(s_genes[9 + offset] - curr_gene[9],2) +
                             pow(s_genes[10 + offset] - curr_gene[10],2) + pow(s_genes[11 + offset] - curr_gene[11],2))+
                             __dsqrt_rz(pow(s_genes[12 + offset] - curr_gene[12],2) + pow(s_genes[13 + offset] - curr_gene[13],2) +
                             pow(s_genes[14 + offset] - curr_gene[14],2) + pow(s_genes[15 + offset] - curr_gene[15],2) +
                             pow(s_genes[16 + offset] - curr_gene[16],2) + pow(s_genes[17 + offset] - curr_gene[17],2));

                results[(threadIdx.x * 32) + j] = dist;
            }

            for(int j = 0; j < 32; j++) {

            	int sharedoffset = j* 32;

                int globaloffset = (blockIdx.x * 32 * geneCount) + (j * geneCount) + (32 * i);

                if(threadIdx.x + globaloffset < geneCount * geneCount) {
            	    distance_d[threadIdx.x + globaloffset ] = results[threadIdx.x + sharedoffset];
                }
            }
        }
    }
}

int main(int argc, char** argv) {

    double* geneList_h, *geneList_d;
    double*  distance_d, *distance_h;

    //Only initialized if calculating all results and GPU is unavailable
    DistanceTuple** distanceMatrix;

    //Program accepts 1 command line argument minimum and 3 maximum
    if(!(argc == 2 || argc == 4 || argc == 6)) {
        printf("Error: Incorrect number of arguments\n");
        usage();
        exit(1);
    }

    //User has requested help, simply prints the usage
    if(strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        usage();
        exit(0);
    }

    //Have one additional command line argument besides the required input file
    if(argc > 3) {
        if(strcmp(argv[2], "-g") == 0 || strcmp(argv[2], "--gene") == 0) {
            selectedGene = argv[3];
            calculateAllGenes = 0;
        } else if (strcmp(argv[2], "-r") == 0 || strcmp(argv[2], "--results") == 0) {
            numberOfResults = atoi(argv[3]);
        } else {
            printf("Error: %s must be either '-g' or '-r'\n", argv[2]);
            usage();
            exit(1);
        }
    } 
    
    if (argc > 5) {
        if(strcmp(argv[4], "-g") == 0 || strcmp(argv[4], "--gene") == 0) {
            selectedGene = argv[5];
            calculateAllGenes = 0;
        } else if (strcmp(argv[4], "-r") == 0 || strcmp(argv[4], "--results") == 0) {
            numberOfResults = atoi(argv[5]);
        } else {
            printf("Error: %s must be either '-g' or '-r'\n", argv[4]);
            usage();
            exit(1);
        }
    }

    //atoi() returns 0 when no valid conversion takes place, need to make sure numberOfResults is at least 1
    if(!(numberOfResults > 0)) {
        printf("Error: number of results must be at least 1. \n");
        exit(1);
    }

    char* filePath = argv[1];

    FILE* file;

    if(!(file = fopen(filePath, "r"))) {
        //File does not exist
        printf("Error: unable to open input file %s\n", filePath);
        exit(1);
    }

    //Allocate memory for the gene list on the host
    geneList_h = (double *) malloc(MAX_GENES * sizeof(double) * REPLICATES_PER_GENE);

    //Allocate memory for the name list on the host
    nameList = (char **) malloc(MAX_GENES * sizeof(char*));

    for(int i = 0; i < MAX_GENES; i++) {
        nameList[i] = (char *) malloc(MAX_NAME_SIZE * sizeof(char));
    }

    //File exists so continue
    createGeneListFromFile(file, geneList_h);

    //Close input file
    fclose(file);

    if(numberOfResults > geneCount - 1) {
        printf("Error: number of results requested  exceeds maximum allowable number.\n");
        printf("Number of genes: %d, maximum number of results per gene: %d\n", geneCount, geneCount-1);
        printf("Number of results will be set to maximum. All results will be saved.\n");
        numberOfResults = geneCount - 1;
    }

    int geneListSize = geneCount * REPLICATES_PER_GENE * sizeof(double);

    printf("Number of genes: %d\n", geneCount);

    if(calculateAllGenes) {
        //Get memory specifications of the GPU
        getCardSpecs();
    }

    //Launch the CUDA portion of the code if calculating distances for all genes

    printf("CUDA status (0 - cannot use, 1 - using): %d\n", canUseCuda);

    if(!canUseCuda) {
        printf("Program will continue in serial mode...\n");
    }

    if(calculateAllGenes && canUseCuda) {

        //There will be n^2 results from n genes
        long long resultsSize = geneCount * geneCount * sizeof(double);

        dim3 blockSize = 32;
        dim3 gridSize = (geneCount % 32 == 0) ? geneCount/32 : geneCount/32 + 1;

        //Allocate space on the host for the distance results
        distance_h = (double*) malloc(resultsSize);

        //Allocate memory on the device for the genelist and distance results
        cudaMalloc((void**) &geneList_d, geneListSize);
        cudaMalloc((void**) &distance_d, resultsSize);

        //Copy the gene list to the device
        cudaMemcpy(geneList_d, geneList_h, geneListSize, cudaMemcpyHostToDevice);

        //Bind geneList to texture memory
        cudaBindTexture(NULL, geneTex, geneList_d, geneListSize);

        calculateDistanceGPU<<<gridSize,blockSize>>>(distance_d, geneCount);

        cudaError_t error = cudaGetLastError();

        if(error != cudaSuccess) {
    	    printf("CUDA error: %s\n", cudaGetErrorString(error));
    	    exit(1);
        }

        //Get results back from the kernel
        cudaMemcpy(distance_h, distance_d, resultsSize, cudaMemcpyDeviceToHost);

        cudaUnbindTexture(geneTex);
        cudaFree(geneList_d);
        cudaFree(distance_d);

        cudaDeviceReset();

    } else if(calculateAllGenes && !canUseCuda) {

        //Allocate memory for the distance matrix
        distanceMatrix = (DistanceTuple**) malloc(geneCount * sizeof(DistanceTuple*));
         
        for(int i = geneCount - 1; i >= 0; i--) {
            distanceMatrix[geneCount - i - 1] = (DistanceTuple*) malloc(i * sizeof(DistanceTuple));
        }

        calculateDistanceCPU(geneList_h, distanceMatrix);

    } else if (!calculateAllGenes){

        distance_h = (double *) malloc(geneCount * sizeof(double));

        calculateSingleDistance(selectedGene, geneList_h, distance_h);
    }

    sortAndPrint(geneList_h, distance_h, distanceMatrix);

    /**************
    |   CLEANUP
    **************/

    free(geneList_h);

    //distance_h is only initialized when using CUDA
    if(calculateAllGenes && canUseCuda) {
        free(distance_h);
    }

    if(calculateAllGenes && !canUseCuda) {
        for(int i = 0; i < geneCount - 1; i++) {
            free(distanceMatrix[i]);
        }

        free(distanceMatrix);
    }

    printf("Done! Results have been stored in the \"output\" folder.\n");
    return 0;
}

void getCardSpecs() {
    int devCount;
    cudaGetDeviceCount(&devCount);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    long long geneListSize = REPLICATES_PER_GENE * 18 * geneCount * sizeof(double);
    long long resultsSize = geneCount * geneCount * sizeof(double);

    if(props.totalGlobalMem == 0) {
        printf("Warning: No CUDA card detected.\n");
        canUseCuda = 0;
        return;
    } else if(props.totalGlobalMem < geneListSize + resultsSize) {
        printf("Warning: CUDA card has insufficient total memory to run for this input size.\n");
        printf("Global memory (MB): %d\n", props.totalGlobalMem/(1024 * 1024));
        canUseCuda = 0;
        return;
    } else if(props.major < 2) {
        printf("Warning: CUDA card compute capability is too low. Expected compute capability is 2.0 or greater. This card has a compute capability of %d.%d\n", props.major, props.minor);
        canUseCuda = 0;
        return;
    }

    canUseCuda = 1;
}

void sortAndPrint(double* geneList, double* distance, DistanceTuple** distanceMatrix) {

    //Need to reconstruct a templist from the distance matrix for each gene
    DistanceTuple* tempDistanceList = (DistanceTuple*) malloc(geneCount * sizeof(DistanceTuple));

    int top = (calculateAllGenes) ? geneCount : 1;

    for(int  i = 0; i < top; i++) {

        if(calculateAllGenes && canUseCuda) {
            for(int j = 0; j < geneCount; j++) {
                tempDistanceList[j].distance = distance[(i*geneCount) + j];
                tempDistanceList[j].geneOne = j;
            }
        } else if (calculateAllGenes && !canUseCuda) {
            
            int row, col, distanceIndex;

            distanceIndex = 0;
            row = i - 1;
            col = 0;

            //Load entires from the diagonal 
            while(row >= 0 && col <= i - 1) {
                tempDistanceList[distanceIndex] = distanceMatrix[row][col];
                distanceIndex++;

                row--;
                col++;
            }

            //Load remaining entries from the gene's column
            for(int j = 0; j < geneCount - i - 1; j++) {
                tempDistanceList[distanceIndex] = distanceMatrix[i][j];
                tempDistanceList[distanceIndex].geneOne = tempDistanceList[distanceIndex].geneTwo;
                distanceIndex++;
            }
        }

        int listSize = (calculateAllGenes && canUseCuda || !calculateAllGenes) ? geneCount : geneCount - 1;

        qsort (tempDistanceList, listSize, sizeof(DistanceTuple), compareDistanceTuple);

        char fname[40];
        strcpy(fname, "output/");
        if(!calculateAllGenes) {
            strcat(fname, selectedGene);
        } else {
            strcat(fname, nameList[i]);
        }
        strcat(fname, ".txt");

        //Write results to file
        FILE * outfile = fopen(fname, "w");

        if(!outfile) {
            printf("Warning: File could not be created. Exiting Program.\n");
            exit(1);
        }

        int startIndex = (calculateAllGenes && canUseCuda || !calculateAllGenes) ? 1 : 0;

        for(int j = startIndex; j < numberOfResults + 1; j++) {
            fprintf(outfile, "%d: %s %f\n", j, nameList[tempDistanceList[j].geneOne], tempDistanceList[j].distance);
        }

        fclose(outfile);
    }
}

void calculateSingleDistance(char* gene, double* geneList, double* distanceList) {

    bool foundGene = false;
    int currGeneIndex;

    for(int i = 0; i < geneCount; i++) {
        if(strcmp(nameList[i],gene) == 0) {
           foundGene = true;
           currGeneIndex = i;
           printf("Found gene %s, now calculating distances.\n", gene);
           break;
        }
    }

    if(!foundGene) {
        printf("Error: unable to find specified gene: %s. \n", gene);
        exit(1);
    }

    double dist = 0.0;

    int currOffset = currGeneIndex * REPLICATES_PER_GENE;

    for(int i = 0; i < geneCount; i++) {

        int tmpOffset = i * REPLICATES_PER_GENE;

        dist = sqrt(pow(geneList[0 + tmpOffset] - geneList[0 + currOffset],2) + pow(geneList[1 + tmpOffset] - geneList[1 + currOffset],2) +
                    pow(geneList[2 + tmpOffset] - geneList[2 + currOffset],2) + pow(geneList[3 + tmpOffset] - geneList[3 + currOffset],2) +
                    pow(geneList[4 + tmpOffset] - geneList[4 + currOffset],2) + pow(geneList[5 + tmpOffset] - geneList[5 + currOffset],2))
             + sqrt(pow(geneList[6 + tmpOffset] - geneList[6 + currOffset],2) + pow(geneList[7 + tmpOffset] - geneList[7 + currOffset],2) +
                    pow(geneList[8 + tmpOffset] - geneList[8 + currOffset],2) + pow(geneList[9 + tmpOffset] - geneList[9 + currOffset],2) +
                    pow(geneList[10 + tmpOffset] - geneList[10 + currOffset],2) + pow(geneList[11 + tmpOffset] - geneList[11 + currOffset],2))
             + sqrt(pow(geneList[12 + tmpOffset] - geneList[12 + currOffset],2) + pow(geneList[13 + tmpOffset] - geneList[13 + currOffset],2) +
                    pow(geneList[14 + tmpOffset] - geneList[14 + currOffset],2) + pow(geneList[15 + tmpOffset] - geneList[15 + currOffset],2) +
                    pow(geneList[16 + tmpOffset] - geneList[16 + currOffset],2) + pow(geneList[17 + tmpOffset] - geneList[17 + currOffset],2));

        distanceList[i] = dist;
    }
}

void trim(char *str) {
    char *p1 = str, *p2 = str;
    do
        while (*p2 == ' ' || *p2 == '/')
            p2++;
        while (*p1++ = *p2++);
}

void calculateDistanceCPU(double* geneList, DistanceTuple** distanceMatrix) {
    
    double dist = 0.0;
    
    int distColIndex, currOffset, tmpOffset, currGeneIndex;

    for(int i = 0; i < geneCount; i++) {
    
        currGeneIndex = i;
    
        currOffset = currGeneIndex * REPLICATES_PER_GENE;
        
        distColIndex = 0;
    
        for(int j = i+1; j < geneCount; j++) {
    
            tmpOffset = j * REPLICATES_PER_GENE;
            
            dist = sqrt(pow(geneList[0 + tmpOffset] - geneList[0 + currOffset],2) + pow(geneList[1 + tmpOffset] - geneList[1 + currOffset],2) +
                        pow(geneList[2 + tmpOffset] - geneList[2 + currOffset],2) + pow(geneList[3 + tmpOffset] - geneList[3 + currOffset],2) +
                        pow(geneList[4 + tmpOffset] - geneList[4 + currOffset],2) + pow(geneList[5 + tmpOffset] - geneList[5 + currOffset],2))
                 + sqrt(pow(geneList[6 + tmpOffset] - geneList[6 + currOffset],2) + pow(geneList[7 + tmpOffset] - geneList[7 + currOffset],2) +
                        pow(geneList[8 + tmpOffset] - geneList[8 + currOffset],2) + pow(geneList[9 + tmpOffset] - geneList[9 + currOffset],2) +
                        pow(geneList[10 + tmpOffset] - geneList[10 + currOffset],2) + pow(geneList[11 + tmpOffset] - geneList[11 + currOffset],2))
                 + sqrt(pow(geneList[12 + tmpOffset] - geneList[12 + currOffset],2) + pow(geneList[13 + tmpOffset] - geneList[13 + currOffset],2) +
                        pow(geneList[14 + tmpOffset] - geneList[14 + currOffset],2) + pow(geneList[15 + tmpOffset] - geneList[15 + currOffset],2) +
                        pow(geneList[16 + tmpOffset] - geneList[16 + currOffset],2) + pow(geneList[17 + tmpOffset] - geneList[17 + currOffset],2));
             
             
            distanceMatrix[i][distColIndex].geneOne = i;
            distanceMatrix[i][distColIndex].geneTwo = j;
            distanceMatrix[i][distColIndex].distance = dist;
             
            distColIndex++;
        }
    }
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
    printf("Please enter the path of the input file as your first command line argument.\n");
    printf("Valid additional arguments include: \n\t -h, --help: displays this usage message\n");
    printf("\t-r, --results: specify the number of results to save for each gene\n");
    printf("\t-g, --gene: specify a specific gene to generate results for\n");
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

void createGeneListFromFile(FILE* file, double* geneList) {

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

            strcpy(nameList[geneCount], tok);

            //Eliminate unwanted characters (spaces, slashes)
            trim(nameList[geneCount]);

            //Populate replicates into geneList
            for(int i = 0; i < REPLICATES_PER_GENE; i++) {
                tok = strtok(NULL,",");

                if(strcmp(tok,"=") == 0) {
                    geneList[i + (geneCount * REPLICATES_PER_GENE)] = 0.0f;
                } else {
                    geneList[i + (geneCount * REPLICATES_PER_GENE)] = atof(tok);
                }
            }
        geneCount++;
        }
    }
}
