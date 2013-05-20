#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cuda.h"
#include <ctype.h>
#include "genedist.h"
#include "cuda_runtime.h"

char** nameList;

int geneCount = 0;

//Determines if the program will run for all genes or a single one
unsigned char singleGeneCalculation = 0;

//May be unable to use CUDA due to GPU memory unavilability
unsigned char canUseCuda;

//User provided parameter, number of results that will be saved for each gene
int numberOfResults = 10;

//User specified gene, set by command line argument (optional)
char selectedGene[MAX_GENE_NAME_SIZE] = "default";

//User specified plates, set by command line argument (optional)
int* selectedPlates = NULL;
int  numSelectedPlates = -1;

//User specified threshold values, set by command line argument(optional)
//Default is -1 which indicates value was not set by user and no threshold
//PPIB will be used (all genes will be included in the calculations)
int highPPIB = -1;
int lowPPIB = -1;

//User specified output file save location
//DEFAULT: "output/"
char outputLocation[MAX_FILE_PATH_SIZE] = "output";

//User specified input file location
char inputLocation[MAX_FILE_PATH_SIZE] = "none";

//Texture memory will be used to store gene information
texture<int2, 1, cudaReadModeElementType> geneTex;

//User specified calculation variant
calculationVariant calcVariant = STANDARD;

__global__ void calculateDistanceGPU(double* distance_d, int geneCount, calculationVariant calcVariant) {

    __shared__ double s_genes[32 * DISTANCES_PER_GENE];
    __shared__ double results[1024];

    double dist = 0.0;

    int geneIndex = blockIdx.x * blockDim.x + threadIdx.x;

    //Fill own gene (these memory accesses are not very much fun but can't be avoided if we use texture memory)
    double curr_gene[DISTANCES_PER_GENE];

	double avgReplicateOne[6];

    if(geneIndex < geneCount) {

        for(int i = 0; i < DISTANCES_PER_GENE; i++) {
        	int2 v = tex1Dfetch(geneTex, geneIndex * DISTANCES_PER_GENE + i);
        	curr_gene[i] = __hiloint2double(v.y, v.x);
        }

		if(calcVariant == AVERAGE_FIRST_DISTANCE) {
    		avgReplicateOne[0] = ((curr_gene[0] + curr_gene[6] + curr_gene[12])/3);
    		avgReplicateOne[1] = ((curr_gene[1] + curr_gene[7] + curr_gene[13])/3);
    		avgReplicateOne[2] = ((curr_gene[2] + curr_gene[8] + curr_gene[14])/3);
    		avgReplicateOne[3] = ((curr_gene[3] + curr_gene[9] + curr_gene[15])/3);
    		avgReplicateOne[4] = ((curr_gene[4] + curr_gene[10] + curr_gene[16])/3);
    		avgReplicateOne[5] = ((curr_gene[5] + curr_gene[11] + curr_gene[17])/3);
        }

        int top = (geneCount % 32 == 0) ? geneCount/32 : geneCount/32 + 1;

        for(int i = 0; i < top; i++) {

            //Fill the shared input array collaboratively 
        	for(int j = 0; j < DISTANCES_PER_GENE; j++) {

        	    int2 v = tex1Dfetch(geneTex, (i * 32 * DISTANCES_PER_GENE) + (threadIdx.x * DISTANCES_PER_GENE) + j);
        	    s_genes[threadIdx.x * DISTANCES_PER_GENE + j] =  __hiloint2double(v.y, v.x);
            }

            for(int j = 0; j < 32; j++) {

            	int offset = DISTANCES_PER_GENE * j;

            	if(calcVariant == STANDARD) {

					dist	=    __dsqrt_rz(pow(s_genes[0 + offset] - curr_gene[0],2) + pow(s_genes[1 + offset] - curr_gene[1],2) +
								 pow(s_genes[2 + offset] - curr_gene[2],2) + pow(s_genes[3 + offset] - curr_gene[3],2) +
								 pow(s_genes[4 + offset] - curr_gene[4],2) + pow(s_genes[5 + offset] - curr_gene[5],2)) +
								 __dsqrt_rz(pow(s_genes[6 + offset] - curr_gene[6],2) + pow(s_genes[7 + offset] - curr_gene[7],2) +
								 pow(s_genes[8 + offset] - curr_gene[8],2) + pow(s_genes[9 + offset] - curr_gene[9],2) +
								 pow(s_genes[10 + offset] - curr_gene[10],2) + pow(s_genes[11 + offset] - curr_gene[11],2))+
								 __dsqrt_rz(pow(s_genes[12 + offset] - curr_gene[12],2) + pow(s_genes[13 + offset] - curr_gene[13],2) +
								 pow(s_genes[14 + offset] - curr_gene[14],2) + pow(s_genes[15 + offset] - curr_gene[15],2) +
								 pow(s_genes[16 + offset] - curr_gene[16],2) + pow(s_genes[17 + offset] - curr_gene[17],2));

            	} 
				
				else if (calcVariant == AVERAGE_FIRST_DISTANCE) {

                	dist	=    __dsqrt_rz(pow(s_genes[0 + offset] - avgReplicateOne[0],2) + pow(s_genes[1 + offset] - avgReplicateOne[1],2) +
                			     pow(s_genes[2 + offset] - avgReplicateOne[2],2) + pow(s_genes[3 + offset] - avgReplicateOne[3],2) +
                                 pow(s_genes[4 + offset] - avgReplicateOne[4],2) + pow(s_genes[5 + offset] - avgReplicateOne[5],2)) +
                                 __dsqrt_rz(pow(s_genes[6 + offset] - avgReplicateOne[0],2) + pow(s_genes[7 + offset] - avgReplicateOne[1],2) +
                                 pow(s_genes[8 + offset] - avgReplicateOne[2],2) + pow(s_genes[9 + offset] - avgReplicateOne[3],2) +
                                 pow(s_genes[10 + offset] - avgReplicateOne[4],2) + pow(s_genes[11 + offset] - avgReplicateOne[5],2))+
                                 __dsqrt_rz(pow(s_genes[12 + offset] - avgReplicateOne[0],2) + pow(s_genes[13 + offset] - curr_gene[1],2) +
                                 pow(s_genes[14 + offset] - avgReplicateOne[2],2) + pow(s_genes[15 + offset] - avgReplicateOne[3],2) +
                                 pow(s_genes[16 + offset] - avgReplicateOne[4],2) + pow(s_genes[17 + offset] - avgReplicateOne[5],2));
				}

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

void parseArguments(int argc, char** argv) {

    if(argc > 1) {

        //Will need to loop through all command line arguments
        for(int i = 1; i < argc; i+=2) {

        	if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) {

        		strcpy(inputLocation, argv[i+1]);

        	} else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
                usage();
                exit(0);
        	} else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--calc") == 0) {

        		//Convert string to upper case
        		char str[MAX_FILE_PATH_SIZE] = "temp";
        		strcpy(str, argv[i+1]);
        		int i = 0;


        		while(i < strlen(str)) {
        			str[i] = (char)toupper(str[i]);
        			i++;
        		}

        		if(strcmp(str, "STANDARD") == 0) {
        			calcVariant = STANDARD;
        		} else if (strcmp(str, "AVERAGE_FIRST_DISTANCE") == 0) {
        			calcVariant = AVERAGE_FIRST_DISTANCE;
        		} else {
        			printf("Warning: No such calculation variant as %s. Please see usage below for a list of acceptable variants.\n", str);
        			usage();
        			exit(1);
        		}

        	} else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--results") == 0) {
  
                numberOfResults = atoi(argv[i+1]);

                //atoi() returns 0 when no valid conversion takes place, need to make sure numberOfResults is at least 1
                if(!(numberOfResults > 0)) {
                    printf("Error: number of results must be at least 1.\n");
                    exit(1);
                }
            } else if (strcmp(argv[i], "-g") == 0 || strcmp(argv[i], "--gene") == 0) {
               
                strcpy(selectedGene,argv[i+1]);
                singleGeneCalculation = 1;

            } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--plates") == 0) {
                //Format will be numbers separated by commas (i.e. 123,233,11,22)

		        //First malloc some space in the selectedPlates array
                //Because size is not know at compile time, make array as big as number of characters in the string
                //This is obviously overkill but will always have enough room for any arbitrary number of plates
                selectedPlates = (int*) malloc(strlen(argv[i+1]) * sizeof(int));

                //If there is only one plate selected, don't need to tokenize the string
                if(strlen(argv[i+1]) == 1) {
                    numSelectedPlates = 1;
                    selectedPlates[0] = atoi(argv[i+1]);
                } else {

                    char* tok = strtok(argv[i+1], ",");
                    numSelectedPlates = 0;

                    while(tok != NULL) {
                        selectedPlates[numSelectedPlates] = atoi(tok);
                        tok = strtok(NULL, ",");
                        numSelectedPlates++;
                    }
                }

            } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--highPPIB") == 0) {

                highPPIB = atoi(argv[i+1]);

                //atoi() returns 0 when on valid conversion takes place, need to make sure threshold PPIB is at least 1
                if(!(highPPIB > 0)) {
                    printf("Error: High threshold PPIB must be at least 1.\n");
                    exit(1);
                }

            } else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--lowPPIB") == 0) {

                lowPPIB = atoi(argv[i+1]);

                //atoi() returns 0 when on valid conversion takes place, need to make sure threshold PPIB is at least 1
                if(!(lowPPIB > 0)) {
                    printf("Error: Low threshold PPIB must be at least 1.\n");
                    exit(1);
                }

            } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0 ){

            	strcpy(outputLocation, argv[i+1]);

				/*********************************************************
				*
				* Not sure if this is necessary, simply adds a trailing '/'
				* Maybe different for windows vs. posix machines as well
				*
				**********************************************************/
            	//if(outputLocation[strlen(outputLocation) - 1] != '/') {
            	//	outputLocation[strlen(outputLocation)] = '/';
            	//}
				
            } else {
                printf("Warning: %s is an invalid command line argument\n. See below for usage.\n\n", argv[i]);
                usage();
                exit(1);
            }
        }
    } else {
    	printf("No command line arguments specified. Please see usage below.\n\n");
    	usage();
    	exit(1);
    }
}

int main(int argc, char** argv) {

    double* geneList_h, *geneList_d;
    double*  distance_d, *distance_h;

    //Only holds data if calculating all results and GPU is unavailable
    DistanceTuple** distanceMatrix = NULL;

	distance_h = NULL;

    parseArguments(argc, argv);
    
    if(strcmp(inputLocation, "none") == 0) {
    	printf("Warning: must provide an input file.\n\n");
    	usage();
    	exit(1);
    }

    FILE* inputFile;

    if(!(inputFile = fopen(inputLocation, "r"))) {
        //File does not exist
        printf("Error: unable to open input file %s\n", inputLocation);
        exit(1);
    }

    //Allocate memory for the gene list on the host
    geneList_h = (double *) malloc(MAX_GENES * sizeof(double) * DISTANCES_PER_GENE);

    //Allocate memory for the name list on the host
    nameList = (char **) malloc(MAX_GENES * sizeof(char*));

    for(int i = 0; i < MAX_GENES; i++) {
        nameList[i] = (char *) malloc(MAX_GENE_NAME_SIZE * sizeof(char));
    }

    //File exists so continue
    createGeneListFromFile(inputFile, geneList_h);

    //Close input file
    fclose(inputFile);

    printf("Read file successfully.\n");

    if(geneCount == 1) {
        printf("Only one gene found that meets specified criteria. Please provide new criteria to expand number of eligible genes.\n");
        exit(0);
    }

    if(geneCount < 11) {
        numberOfResults = geneCount - 2;
    }

    if(numberOfResults > geneCount - 1) {
        printf("Error: number of results requested  exceeds maximum allowable number.\n");
        printf("Number of genes: %d, maximum number of results per gene: %d\n", geneCount, geneCount-1);
        printf("Number of results will be set to maximum. All results will be saved.\n");
        numberOfResults = geneCount - 1;
    }

    int geneListSize = geneCount * DISTANCES_PER_GENE * sizeof(double);

    printf("Number of genes: %d\n", geneCount);

    if(!singleGeneCalculation) {
        //Get memory specifications of the GPU
        getCardSpecs();
    }

    if(geneCount < CUDA_CUTOFF) {
        canUseCuda = 0;
    }

    //Launch the CUDA portion of the code if calculating distances for all genes
    printf("CUDA status (0 - not using, 1 - using): %d\n", canUseCuda);

    if(!canUseCuda) {
        printf("Program will continue in serial mode...\n");
    }

    if(!singleGeneCalculation && canUseCuda) {

		cudaError_t error;

        //There will be n^2 results from n genes
        long long resultsSize = geneCount * geneCount * sizeof(double);

        dim3 blockSize = 32;
        dim3 gridSize = 0;     
 
        gridSize = (geneCount % 32 == 0) ? geneCount/32 : geneCount/32 + 1;
        
		error = cudaMalloc((void**) &distance_d, resultsSize);
		checkCudaError(error, "Malloc for device side distance array");

        //Allocate space on the host for the distance results
        distance_h = (double*) malloc(resultsSize);

        //Allocate memory on the device for the genelist
        error = cudaMalloc((void**) &geneList_d, geneListSize);
		checkCudaError(error, "Malloc for device side gene list array");

        //Copy the gene list to the device
        error = cudaMemcpy(geneList_d, geneList_h, geneListSize, cudaMemcpyHostToDevice);
		checkCudaError(error, "Copy genelist from host to device");

        //Bind geneList to texture memory
        error = cudaBindTexture(NULL, geneTex, geneList_d, geneListSize);
		checkCudaError(error, "Bind genelist to texture memory");

        calculateDistanceGPU<<<gridSize,blockSize>>>(distance_d, geneCount, calcVariant);

		error = cudaGetLastError();
        
		//If there is an error in the kernel, exit the program because results will be invalid
		if(error) {
			printf("Progrm experienced error in kernel call and will now exit.\n");
    	    exit(1);
        }

        //Get results back from the kernel
        error = cudaMemcpy(distance_h, distance_d, resultsSize, cudaMemcpyDeviceToHost);
		checkCudaError(error, "Copy results from Device to Host");

        error = cudaUnbindTexture(geneTex);
        checkCudaError(error, "Unbind Texture");

        error = cudaFree(geneList_d);
        checkCudaError(error, "Free genelist");

        error = cudaFree(distance_d);
        checkCudaError(error, "Free distance_d");

        error = cudaDeviceReset();
        checkCudaError(error, "Device Reset");

    } else if(!singleGeneCalculation && !canUseCuda) {

        //Allocate memory for the distance matrix
        distanceMatrix = (DistanceTuple**) malloc(geneCount * sizeof(DistanceTuple*));
         
        for(int i = geneCount - 1; i >= 0; i--) {
            distanceMatrix[geneCount - i - 1] = (DistanceTuple*) malloc(i * sizeof(DistanceTuple));
        }

        calculateDistanceCPU(geneList_h, distanceMatrix);

    } else if (singleGeneCalculation){

        distance_h = (double *) malloc(geneCount * sizeof(double));

        calculateSingleDistance(selectedGene, geneList_h, distance_h);
    }

    sortAndPrint(geneList_h, distance_h, distanceMatrix);

    /**************
    |   CLEANUP
    **************/

    free(geneList_h);

    if((!singleGeneCalculation && canUseCuda) || singleGeneCalculation) {
        free(distance_h);
    }

    if(!singleGeneCalculation && !canUseCuda) {
        for(int i = 0; i < geneCount - 1; i++) {
            free(distanceMatrix[i]);
        }

        free(distanceMatrix);
    }

    printf("Done! Results have been stored in the \"%s\" folder.\n", outputLocation);
    return 0;
}

bool checkCudaError(cudaError_t error, char* operation) {

	if(strcmp(cudaGetErrorString(error), "no error") != 0) {

		printf("Warning: the following CUDA error occurred: %s\n", cudaGetErrorString(error));
		printf("This error occurred during the following operation: %s\n\n", operation);

		return 1;
	}

	return 0;
}

void getCardSpecs() {
    int devCount;
    cudaGetDeviceCount(&devCount);

    size_t freemem, totmem;


    cudaMemGetInfo(&freemem, &totmem);

    printf("Total mem: %d\n", totmem);
    printf("Free mem:  %d\n", freemem);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    long long geneListSize = DISTANCES_PER_GENE * REPLICATES_PER_GENE * geneCount * sizeof(double);
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

    int top = (!singleGeneCalculation) ? geneCount : 1;

    for(int  i = 0; i < top; i++) {

        if((!singleGeneCalculation && canUseCuda) || singleGeneCalculation) {
            for(int j = 0; j < geneCount; j++) {
                tempDistanceList[j].distance = distance[(i*geneCount) + j];
                tempDistanceList[j].geneOne = j;
            }
        } else if (!singleGeneCalculation && !canUseCuda) {
            
            int row, col, distanceIndex;

            distanceIndex = 0;
            row = i - 1;
            col = 0;

            //Load entries from the diagonal
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

        int listSize = ( (!singleGeneCalculation && canUseCuda)|| singleGeneCalculation) ? geneCount : geneCount - 1;

        qsort (tempDistanceList, listSize, sizeof(DistanceTuple), compareDistanceTuple);

        char fname[MAX_FILE_PATH_SIZE + MAX_GENE_NAME_SIZE];

        strcpy(fname, outputLocation);
        if(singleGeneCalculation) {
            strcat(fname, selectedGene);
        } else {
            strcat(fname, nameList[i]);
        }
        strcat(fname, ".txt");

        //Write results to file
        FILE * outfile = fopen(fname, "w");

        if(!outfile) {
            printf("Warning: Output folder does not exist. Please create the output folder before running the program.\n");
            exit(1);
        }

        int startIndex = ((!singleGeneCalculation && canUseCuda) || singleGeneCalculation) ? 1 : 0;

        for(int j = startIndex; j < numberOfResults + startIndex; j++) {
        	if(strcmp(nameList[tempDistanceList[j].geneOne], nameList[i]) != 0) {
        		fprintf(outfile, "%d: %s %.15f\n", startIndex ? j : j+1, nameList[tempDistanceList[j].geneOne], tempDistanceList[j].distance);
        	}
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

    int currOffset = currGeneIndex * DISTANCES_PER_GENE;

	double avgReplicateOne[6];

	if(calcVariant == AVERAGE_FIRST_DISTANCE) {
		avgReplicateOne[0] = ((geneList[0 + currOffset] + geneList[6 + currOffset] + geneList[12 + currOffset])/3);
		avgReplicateOne[1] = ((geneList[1 + currOffset] + geneList[7 + currOffset] + geneList[13 + currOffset])/3);
		avgReplicateOne[2] = ((geneList[2 + currOffset] + geneList[8 + currOffset] + geneList[14 + currOffset])/3);
		avgReplicateOne[3] = ((geneList[3 + currOffset] + geneList[9 + currOffset] + geneList[15 + currOffset])/3);
		avgReplicateOne[4] = ((geneList[4 + currOffset] + geneList[10 + currOffset] + geneList[16 + currOffset])/3);
		avgReplicateOne[5] = ((geneList[5 + currOffset] + geneList[11 + currOffset] + geneList[17 + currOffset])/3);
	}

    for(int i = 0; i < geneCount; i++) {

        int tmpOffset = i * DISTANCES_PER_GENE;

		if(calcVariant == STANDARD) {

			dist = sqrt(pow(geneList[0 + tmpOffset] - geneList[0 + currOffset],2) + pow(geneList[1 + tmpOffset] - geneList[1 + currOffset],2) +
						pow(geneList[2 + tmpOffset] - geneList[2 + currOffset],2) + pow(geneList[3 + tmpOffset] - geneList[3 + currOffset],2) +
						pow(geneList[4 + tmpOffset] - geneList[4 + currOffset],2) + pow(geneList[5 + tmpOffset] - geneList[5 + currOffset],2))
				 + sqrt(pow(geneList[6 + tmpOffset] - geneList[6 + currOffset],2) + pow(geneList[7 + tmpOffset] - geneList[7 + currOffset],2) +
						pow(geneList[8 + tmpOffset] - geneList[8 + currOffset],2) + pow(geneList[9 + tmpOffset] - geneList[9 + currOffset],2) +
						pow(geneList[10 + tmpOffset] - geneList[10 + currOffset],2) + pow(geneList[11 + tmpOffset] - geneList[11 + currOffset],2))
				 + sqrt(pow(geneList[12 + tmpOffset] - geneList[12 + currOffset],2) + pow(geneList[13 + tmpOffset] - geneList[13 + currOffset],2) +
						pow(geneList[14 + tmpOffset] - geneList[14 + currOffset],2) + pow(geneList[15 + tmpOffset] - geneList[15 + currOffset],2) +
						pow(geneList[16 + tmpOffset] - geneList[16 + currOffset],2) + pow(geneList[17 + tmpOffset] - geneList[17 + currOffset],2));

        } else if(calcVariant == AVERAGE_FIRST_DISTANCE) {

			dist = sqrt(pow(geneList[0 + tmpOffset] - avgReplicateOne[0],2) + pow(geneList[1 + tmpOffset] - avgReplicateOne[1],2) +
						pow(geneList[2 + tmpOffset] - avgReplicateOne[2],2) + pow(geneList[3 + tmpOffset] - avgReplicateOne[3],2) +
						pow(geneList[4 + tmpOffset] - avgReplicateOne[4],2) + pow(geneList[5 + tmpOffset] - avgReplicateOne[5],2))
				 + sqrt(pow(geneList[6 + tmpOffset] - avgReplicateOne[0],2) + pow(geneList[7 + tmpOffset] - avgReplicateOne[1],2) +
						pow(geneList[8 + tmpOffset] - avgReplicateOne[2],2) + pow(geneList[9 + tmpOffset] - avgReplicateOne[3],2) +
						pow(geneList[10 + tmpOffset] - avgReplicateOne[4],2) + pow(geneList[11 + tmpOffset] - avgReplicateOne[5],2))
				 + sqrt(pow(geneList[12 + tmpOffset] - avgReplicateOne[0],2) + pow(geneList[13 + tmpOffset] - avgReplicateOne[1],2) +
						pow(geneList[14 + tmpOffset] - avgReplicateOne[2],2) + pow(geneList[15 + tmpOffset] - avgReplicateOne[3],2) +
						pow(geneList[16 + tmpOffset] - avgReplicateOne[4],2) + pow(geneList[17 + tmpOffset] - avgReplicateOne[5],2));
        }

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
    
        currOffset = currGeneIndex * DISTANCES_PER_GENE;

		double avgReplicateOne[6];

    	if(calcVariant == AVERAGE_FIRST_DISTANCE) {
    		avgReplicateOne[0] = ((geneList[0 + currOffset] + geneList[6 + currOffset] + geneList[12 + currOffset])/3);
    		avgReplicateOne[1] = ((geneList[1 + currOffset] + geneList[7 + currOffset] + geneList[13 + currOffset])/3);
    		avgReplicateOne[2] = ((geneList[2 + currOffset] + geneList[8 + currOffset] + geneList[14 + currOffset])/3);
    		avgReplicateOne[3] = ((geneList[3 + currOffset] + geneList[9 + currOffset] + geneList[15 + currOffset])/3);
    		avgReplicateOne[4] = ((geneList[4 + currOffset] + geneList[10 + currOffset] + geneList[16 + currOffset])/3);
    		avgReplicateOne[5] = ((geneList[5 + currOffset] + geneList[11 + currOffset] + geneList[17 + currOffset])/3);
    	}
        
        distColIndex = 0;
    
        for(int j = i+1; j < geneCount; j++) {
    
            tmpOffset = j * DISTANCES_PER_GENE;

			if(calcVariant == STANDARD) {

            	 dist = sqrt(pow(geneList[0 + tmpOffset] - geneList[0 + currOffset],2) + pow(geneList[1 + tmpOffset] - geneList[1 + currOffset],2) +
							pow(geneList[2 + tmpOffset] - geneList[2 + currOffset],2) + pow(geneList[3 + tmpOffset] - geneList[3 + currOffset],2) +
							pow(geneList[4 + tmpOffset] - geneList[4 + currOffset],2) + pow(geneList[5 + tmpOffset] - geneList[5 + currOffset],2))
					 + sqrt(pow(geneList[6 + tmpOffset] - geneList[6 + currOffset],2) + pow(geneList[7 + tmpOffset] - geneList[7 + currOffset],2) +
							pow(geneList[8 + tmpOffset] - geneList[8 + currOffset],2) + pow(geneList[9 + tmpOffset] - geneList[9 + currOffset],2) +
							pow(geneList[10 + tmpOffset] - geneList[10 + currOffset],2) + pow(geneList[11 + tmpOffset] - geneList[11 + currOffset],2))
					 + sqrt(pow(geneList[12 + tmpOffset] - geneList[12 + currOffset],2) + pow(geneList[13 + tmpOffset] - geneList[13 + currOffset],2) +
							pow(geneList[14 + tmpOffset] - geneList[14 + currOffset],2) + pow(geneList[15 + tmpOffset] - geneList[15 + currOffset],2) +
							pow(geneList[16 + tmpOffset] - geneList[16 + currOffset],2) + pow(geneList[17 + tmpOffset] - geneList[17 + currOffset],2));
            } else if (calcVariant == AVERAGE_FIRST_DISTANCE) {

				dist = sqrt(pow(geneList[0 + tmpOffset] - avgReplicateOne[0],2) + pow(geneList[1 + tmpOffset] - avgReplicateOne[1],2) +
							pow(geneList[2 + tmpOffset] - avgReplicateOne[2],2) + pow(geneList[3 + tmpOffset] - avgReplicateOne[3],2) +
							pow(geneList[4 + tmpOffset] - avgReplicateOne[4],2) + pow(geneList[5 + tmpOffset] - avgReplicateOne[5],2))
					 + sqrt(pow(geneList[6 + tmpOffset] - avgReplicateOne[0],2) + pow(geneList[7 + tmpOffset] - avgReplicateOne[1],2) +
							pow(geneList[8 + tmpOffset] - avgReplicateOne[2],2) + pow(geneList[9 + tmpOffset] - avgReplicateOne[3],2) +
							pow(geneList[10 + tmpOffset] - avgReplicateOne[4],2) + pow(geneList[11 + tmpOffset] - avgReplicateOne[5],2))
					 + sqrt(pow(geneList[12 + tmpOffset] - avgReplicateOne[0],2) + pow(geneList[13 + tmpOffset] - avgReplicateOne[1],2) +
							pow(geneList[14 + tmpOffset] - avgReplicateOne[2],2) + pow(geneList[15 + tmpOffset] - avgReplicateOne[3],2) +
							pow(geneList[16 + tmpOffset] - avgReplicateOne[4],2) + pow(geneList[17 + tmpOffset] - avgReplicateOne[5],2));
            }
                        
             
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
    printf("User MUST provide an input file path\n\n");
    printf("\t-c, --calc: specify the calculation variant to be run. Valid options are as follows:\n");
    printf("\t\tSTANDARD - dist(r1,s1) + dist(r2,s2) + dist(r3,s3)\n");
    printf("\t\tAVERAGE_FIRST_DISTANCE - dist(avg(r1,r2,r3), s1) + dist(avg(r1,r2,r3), s2) + dist(avg(r1,r2,r3), s3)\n\n");
    printf("\t-g, --gene: specify a specific gene to generate results for\n\n");
    printf("\t-h, --help: displays this usage message\n\n");
    printf("\t-i, --input: specify the input file path\n\n");
    printf("\t-l, --lowPPIB: any genes with an average PPIB value lower than the number specified here will not be included in the calculations\n\n");
    printf("\t-o, --output: specify which folder the output results will be saved in - must already exist\n\n");
    printf("\t-p, --plates: specify which specific plates will be included in the calculations. Format is: plate,plate,plate e.g. '-p 1,5,6' will calculate for only plates 1, 5, and 6.\n\n");
    printf("\t-r, --results: specify the number of results to save for each gene\n\n");
    printf("\t-t, --highPPIB: any genes with an average PPIB value higher than the number specified here will not be included in the calculations\n\n");
}

unsigned char isSelectedPlate(int plateNumber) {

    for(int i = 0; i < numSelectedPlates; i++) {
        if(selectedPlates[i] == plateNumber) {
            return 1;
        }
    }

    return 0;
}

void createGeneListFromFile(FILE* file, double* geneList) {

    char line[512];
    int ppib;

    //Eat the header line
    fgets( line, sizeof(line), file);

    while(fgets( line, sizeof(line),  file ) != NULL)
    {
        char* tok;
        int len = strlen(line);

        //Reset PPIB value for each gene
        ppib = 0;

        if(len > 0 && line[len-1]=='\n'){
            line[len-1] = '\0';
        }
        if(len > 2 && line[len-2]=='\r') {
            line[len-2] = '\0';
        }
        tok = strtok(line,",");

        //Check to ensure tok is not null and that it is not a control
        if(tok && strcmp(tok, "FALSE") == 0) {

            //Get the plate number (next token)
            tok = strtok(NULL,",");

            //check to ensure the plate number is wanted for this calculation
            if(numSelectedPlates != -1 && !isSelectedPlate(atoi(tok))) {
                goto out;
            }

            //Get the Gene Name (next token)
            tok = strtok(NULL,",");

            //Store the name of the gene in the nameList
            strcpy(nameList[geneCount], tok);

            //Eliminate unwanted characters (spaces, slashes)
            trim(nameList[geneCount]);

            //Populate distances into geneList
            for(int i = 0; i < REPLICATES_PER_GENE; i++) {

                for(int j = 0; j < PROBES_PER_REPLICATE; j++) {
                    tok = strtok(NULL,",");

                    if(strcmp(tok,"=") == 0 || strcmp(tok,"#N/A") == 0) {
                        
                        //Break out of nested loops, if one replicate of a gene is invalid, the entire gene is invalid
                        //There is no need to parse any more of this line
                        goto out;

                    } else {
                        geneList[(geneCount * DISTANCES_PER_GENE) + (PROBES_PER_REPLICATE * i) + j] = atof(tok);
                    }
                }
                //Read the PPIB value, add to ppib total
                ppib += atoi(strtok(NULL, ","));
            }
            
            //Calculate the average of the PPIBs for each replicate
            ppib /= 3;

            //Remove a gene that has a sub-threshold PPIB (first check to see that a threshold PPIB has been set)
            if((lowPPIB != -1 && ppib < lowPPIB) || (highPPIB != -1 && ppib > highPPIB)) {
               goto out;
            }

	        geneCount++;

            out:
                continue;
        }
    }
}
