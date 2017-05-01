#include <stdio.h>

#define HEADER_SIZE 44

/* Increases the volume of a .wav file (specified by argv[1]),
   and saves the altered version as a file whose name is
   specified by argv[2]. */
int main(int argc, char *argv[]) {
    char *input_name, *output_name;
    FILE *input_wav, *output_wav;
    short sample;
    short header[HEADER_SIZE];
    int error;
  
    if (argc != 3) {
        fprintf(stderr, "Usage: %s inputfile outputfile\n", argv[0]);
        return 1;
    }

    input_name = argv[1];
    output_name = argv[2];
  
    input_wav = fopen(input_name, "rb");
    if (input_wav == NULL) {
        fprintf(stderr, "Error: could not open input file\n");
        return 1;
    }

    output_wav = fopen(output_name, "wb");
    if (output_wav == NULL) {
        fprintf(stderr, "Error: could not open output file\n");
        return 1;
    }

    fread(header, HEADER_SIZE, 1, input_wav);
    error = fwrite(header, HEADER_SIZE, 1, output_wav); 
    if (error != 1) {
        fprintf(stderr, "Error: could not write a full audio header\n");
        return 1;
    }

    while (fread(&sample, sizeof(short), 1, input_wav) == 1) {
        sample = sample * 4;
        error = fwrite(&sample, sizeof(short), 1, output_wav);
        if (error != 1) {
            fprintf(stderr, "Error: could not write a sample\n");
            return 1;
        }
    }

    error = fclose(input_wav);
    if (error != 0) {
        fprintf(stderr, "Error: fclose failed on input file\n");
        return 1;
    }

    error = fclose(output_wav);
    if (error != 0) {
        fprintf(stderr, "Error: fclose failed on output file\n");
        return 1;
    }

    return 0;
}