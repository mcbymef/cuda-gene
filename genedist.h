#define MAX_GENES 33000 
#define REPLICATES_PER_GENE 18
#define MAX_NAME_SIZE 20

typedef struct __align__(16) {
    unsigned int geneOne;
    unsigned int geneTwo;
    double distance;
} DistanceTuple;

void getCardSpecs();

bool isValidName(char* name);

void trim(char* str);

void usage(void);

void createGeneListFromFile(FILE* file, double* geneList);

void calculateSingleDistance(char* gene, double* geneList, double* distanceList);

void calculateDistanceCPU(double* geneList, DistanceTuple** distanceMatrix);

void sortAndPrint(double* geneList, double* distance, DistanceTuple** distanceMatrix);

int compareDistanceTuple (const void* a, const void* b);
