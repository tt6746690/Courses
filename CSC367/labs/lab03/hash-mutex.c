// ------------
// This code is provided solely for the personal and private use of
// students taking the CSC367 course at the University of Toronto.
// Copying for purposes other than this use is expressly prohibited.
// All forms of distribution of this code, whether as given or with
// any changes, are expressly prohibited.
//
// Authors: Bogdan Simion, Maryam Dehnavi, Alexey Khrabrov
//
// All of the files in this directory and all subdirectories are:
// Copyright (c) 2019 Bogdan Simion and Maryam Dehnavi
// -------------

#include <pthread.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#include "hash.h"


struct node {
    int key;
    int value; 
    struct node* next;      
};
typedef struct node node_t;

struct _hash_table_t {
	int size;// should be a prime number
        node_t** list;
        pthread_mutex_t* mutex;
};

int hash32shift(int key)
{
        key = ~key + (key << 15);
        key = key ^ (key >> 12);
        key = key + (key << 2);
        key = key ^ (key >> 4);
        key = key * 2057;
        key = key ^ (key >> 16);
        return key;
}

void linkedlist_destroy(node_t** head) {

        if (!*head) return;

        node_t* cur = *head;
        node_t* next;
        
        while (cur != NULL) {
                next = cur->next; 
                free(cur);
                cur = next;
        }
        *head = NULL;
}

// returns -1 if not found
int linkedlist_find(node_t**head, int key) {
        if (!*head) return -1;
        for (node_t* cur = *head, *next; cur ; cur = next) {
                if (cur->key == key) 
                        return cur->value;
                next = cur->next;
        }
        return -1;
}

void linkedlist_prepend(node_t**head, int key, int value) {
        node_t* node = (node_t*) malloc(sizeof(node_t));
        node->key = key;
        node->value = value;
        node->next = NULL;
        if (!*head) {
                *head = node;
        } else {
                node_t* next = *head;
                *head = node;
                (*head)->next = next;
        }
}



static bool is_prime(int n)
{
	assert(n > 0);
	for (int i = 2; i <= sqrt(n); i++) {
		if (n % i == 0) return false;
	}
	return true;
}

// Get the smallest prime number that is not less than n (for hash table size computation)
int next_prime(int n)
{
	for (int i = n; ; i++) {
		if (is_prime(i)) return i;
	}
	assert(false);
	return 0;
}


// Create a hash table with 'size' buckets; the storage is allocated dynamically using malloc(); returns NULL on error
hash_table_t *hash_create(int size)
{
	assert(size > 0);
        hash_table_t* table = (hash_table_t*) malloc(sizeof(hash_table_t));
        table->size = size;
        table->list = (node_t**) malloc(size * sizeof(node_t*));
        table->mutex = (pthread_mutex_t*) malloc(size * sizeof(pthread_mutex_t));
        for (int i = 0; i < size; ++i) {
                table->list[i] = NULL;
                pthread_mutex_init(&(table->mutex[i]), NULL);
        }
        return table;
}

// Release all memory used by the hash table, its buckets and entries
void hash_destroy(hash_table_t *table)
{
	assert(table != NULL);
        for (int i = 0; i < table->size; ++i) {
                linkedlist_destroy(&(table->list[i]));
                pthread_mutex_destroy(&(table->mutex[i]));
        }
        free(table->list);
}


// Returns -1 if key is not found
int hash_get(hash_table_t *table, int key)
{
	assert(table != NULL);
        int hash = hash32shift(key) % table->size;

        if (table->list[hash] == NULL) {
                return -1;
        }

        pthread_mutex_lock(&(table->mutex[hash]));
        int value = linkedlist_find(&(table->list[hash]), key);
        pthread_mutex_unlock(&(table->mutex[hash]));

        return value;
}

// Returns 0 on success, -1 on failure
int hash_put(hash_table_t *table, int key, int value)
{
	assert(table != NULL);
        int hash = hash32shift(key) % table->size;

        pthread_mutex_lock(&(table->mutex[hash]));
        linkedlist_prepend(&(table->list[hash]), key, value);
        pthread_mutex_unlock(&(table->mutex[hash]));

	return 0;
}
