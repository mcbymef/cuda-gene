#define MAX_GENES 33000 
#define REPLICATES_PER_GENE 3 
#define PROBES_PER_REPLICATE 6
#define DISTANCES_PER_GENE (REPLICATES_PER_GENE * PROBES_PER_REPLICATE)
#define MAX_NAME_SIZE 20

typedef struct __align__(16) {
    unsigned int geneOne;
    unsigned int geneTwo;
    double distance;
} DistanceTuple;

void calculateSingleDistance(char* gene, double* geneList, double* distanceList);

void calculateDistanceCPU(double* geneList, DistanceTuple** distanceMatrix);

int compareDistanceTuple (const void* a, const void* b);

void createGeneListFromFile(FILE* file, double* geneList);

void getCardSpecs();

bool isValidName(char* name);

void sortAndPrint(double* geneList, double* distance, DistanceTuple** distanceMatrix);

void trim(char* str);

void usage(void);
