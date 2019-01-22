# ------------
# This code is provided solely for the personal and private use of
# students taking the CSC367 course at the University of Toronto.
# Copying for purposes other than this use is expressly prohibited.
# All forms of distribution of this code, whether as given or with
# any changes, are expressly prohibited.
#
# Authors: Bogdan Simion, Maryam Dehnavi, Alexey Khrabrov
#
# All of the files in this directory and all subdirectories are:
# Copyright (c) 2019 Bogdan Simion and Maryam Dehnavi
# -------------

# If this variable is set to 1 (e.g. make is invoked as 'EMBED_DATA=1 make'), the data is embedded into the executable
# instead of loading it from a separate file at runtime (which can help achieve more stable perf measurements)
EMBED_DATA ?= 0
ifeq ($(EMBED_DATA),1)
DATA_OBJ = part2_data.o
endif

CC = gcc
CFLAGS += -std=gnu11 -Wall -Werror -g3 -DEMBED_DATA=$(EMBED_DATA) -DNDEBUG

all: part2 part2-parallel part2-parallel-opt
ifneq ($(EMBED_DATA),1)
all: part2data
endif

data.o: data.h
options.o: options.h
part2_data.o: data.h
part2: time_util.h
part2-parallel: time_util.h
part2-parallel-opt: time_util.h

part2_data_gen: part2_data_gen.o options.o
	$(CC) $^ -o $@

# Generate embedded data with default parameters; add arguments to './part2_data_gen' to change parameters
part2_data.c: part2_data_gen
	./part2_data_gen

datagen: datagen.o data.o options.o
	$(CC) $^ -o $@

# Generate file data with default parameters; add arguments to './datagen' to change parameters
part2data: datagen
	./datagen

part2: part2.o data.o $(DATA_OBJ)
	$(CC) $^ -o $@

part2-parallel: part2-parallel.o data.o $(DATA_OBJ)
	$(CC) $^ -o $@ -pthread

part2-parallel-opt: part2-parallel-opt.o data.o $(DATA_OBJ)
	$(CC) $^ -o $@ -pthread

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o part2_data.c part2data* part2_data_gen datagen part2 part2-parallel part2-parallel-opt
