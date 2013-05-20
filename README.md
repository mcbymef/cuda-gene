INTRODUCTION
-------------
The GeneDistance program will calculate the distance between all genes and rank them in ascending order (smallest distance will be ranked number 1). The calculations are done on CUDA when a capable CUDA GPU is available. To achieve the required precision, the CUDA card must have a compute capability of 2.0 or higher. The program will output a file for each gene that matches the specified filtering criteria (more on this in the 'TO RUN' section) that contains the the top 10 results by default. The source code for the program is written in C, with the CUDA extended language as well, and is primarily intended for use on a Linux system. The inclued Makefile assumes a Linux system. 

TO COMPILE
--------------------
Ensure that you have the 'nvcc' command in your PATH (this is generally taken care of by installing the NVidia CUDA Toolkit). In the folder containing the source code, simply type 'make' and the executable file will be generated.

TO RUN
---------------------
To run the program, user must supply an input file. This is done using the "-i" or "--input" command line argument. For example:

$./GeneDistance -i input.csv

The input file must be in the format and saved as a comma separated value (csv) file. By default, the program saves the output files in the directory "output". A specific output directory can be set using the "-o" or "--output" command like argument. For example:

$./GeneDistance -i input.csv -o ~/Destkop/may_20_output


COMMAND LINE OPTIONS
---------------------
Command line options for this program are as follows:

-c or --calc: Specify a calculation variant to be used. Valid options include the following:
	STANDARD - dist(r1,s1) + dist(r2,s2) + dist(r3,s3)
	AVERAGE_FIRST_DISTANCE - dist(avg(r1,r2,r3), s1) + dist(avg(r1,r2,r3), s2) + dist(avg(r1,r2,r3), s3)
	
	Example: $./GeneDistance -i input.csv -c average_first_distance
	
-g or --gene: Generate output only for a specific gene

	Example: $./GeneDistance -i input.csv -g DYRK1A
	
-h or --help: Display usage for all of the command line arguments and basic information about the program

	Example: $./GeneDistance -h

-i or --input: Specify the input file

	Example: $./GeneDistance -i input.csv

-l or --lowPPIB: Specify a low PPIB threshold. Any genes with an average PPIB less than this value will be removed from the calculations

	Example: $./GeneDistance -i input.csv -l 2500
	
-o or --output: Specify an output directory. Default is "output"

	Example: $./GeneDistance -i input.csv -o ~/Desktop/may_20_output
	
-p or --plates: Specify which plates will be included in the calculation. Format is: plate#,plate#,plate#, ...

	Example: $./GeneDistance -i input.csv -p 1,5,6,7
	This will run calculations ONLY for gene included in plates 1,5,6 or 7
	
-r or --results: Specify the number of results for each gene. Default is 10, minimum is 1, maximum is the total number of genes
	
	Example: $./GeneDistance -i input.csv -r 50
	
-t or --highPPIB: Specify a high PPIB threshold, Any genes with an average PPIB greater than this value will be removed from the calculations

	Example: $./GeneDistance -i input.csv -t 17500
	
All of these command line arguments can be used together to get granular filtering at many levels. For example, the following command:

$./GeneDistance -i input.csv -o ~/Desktop/test_output -l 1750 -t 21000 -p 1,5,6,8,13,16,17 -r 30 -c average_first_distance

will run calculations using the "average_first_distance" calculation variant on genes with an average PPIB between 1750 and 21000 that are also from plates 1,5,6,8,13,16 or 17. The program will save the top 30 results for each gene in the "~/Desktop/test_output" directory.

FILES INCLUDED
---------------------
genedist.cu - main source code for program
genedist.h - header file for main source code
Makefile - makefile to create executable for program
gt - "ground truth" directory containing what are assumed to be correct values for calculations. Use to check against results after modifying code to ensure correctness has not been affected.
serialfiles - directory containing original serial versions of the code
input.csv - standard input file in the correct format. All other input files must follow this format EXACTLY