#define MAX_GENES 33000 
#define REPLICATES_PER_GENE 3 
#define PROBES_PER_REPLICATE 6
#define DISTANCES_PER_GENE (REPLICATES_PER_GENE * PROBES_PER_REPLICATE)
#define MAX_NAME_SIZE 50

//Input size less than this will be run in serial mode
#define CUDA_CUTOFF 4000

//Value above which which CUDA work will be split among two kernels
//An input size of 24000 genes will fit comfortably on 5GB of memory
#define TWO_KERNEL_THRESHOLD 24000

typedef struct __align__(16) {
    unsigned int geneOne;
    unsigned int geneTwo;
    double distance;
} DistanceTuple;

enum calculationVariant { 
	STANDARD,					//dist(r1,s1) + dist(r2,s2) + dist(r3,s3)
    AVERAGE_FIRST_REPLICATE		//dist(avg(r1,r2,r3), s1) + dist(avg(r1,r2,r3), s2) + dist(avg(r1,r2,r3), s3)
};

void calculateSingleDistance(char* gene, double* geneList, double* distanceList);

void calculateDistanceCPU(double* geneList, DistanceTuple** distanceMatrix);

int compareDistanceTuple (const void* a, const void* b);

void createGeneListFromFile(FILE* file, double* geneList);

double calculateStandardVariant(double* geneList, int currGeneOffset, int tmpGeneOffset);

double calculateAverageFirstReplicateVariant(double* geneList, int tmpGeneOffset, double* avgReplicateOne);

void checkCudaError(cudaError_t error, char* operation);

void getCardSpecs();

bool isValidName(char* name);

void sortAndPrint(double* geneList, double* distance, DistanceTuple** distanceMatrix);

void trim(char* str);

void usage(void);
