/* ------------
 * This code is provided solely for the personal and private use of 
 * students taking the CSC367 course at the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited. 
 * All forms of distribution of this code, whether as given or with 
 * any changes, are expressly prohibited. 
 * 
 * Authors: Bogdan Simion, Maryam Dehnavi, Felipe de Azevedo Piovezan
 * 
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2019 Bogdan Simion and Maryam Dehnavi
 * -------------
*/

#ifndef CPU_DOT_H
#define CPU_DOT_H

float dotCPU(float *array1, float *array2, int num_elem) {
	float dot = 0;

	for(int i = 0; i < num_elem; i++)
    {
        dot += array1[i] * array2[i];
	}

	return dot;
}

#endif
