#ifndef UTILITIES_H
#define UTILITIES_H

#include "ext2.h"

typedef struct ext2_inode *ino_ptr;
typedef struct ext2_dir_entry_2 *dir_entry_ptr;

/* important blocks */
unsigned char *disk;

/* first 11 inodes reserved */
#define EXT2_FIRST_NONRESERVED_INO 12

/**
 * For this assignment, assume
 *  32  inodes 
 *  128 blocks
 */
#define INODE_TOTAL 32
#define BLOCK_TOTAL 128

/**
 * @brief	Returns i-th most significant bits of char c (little endian)
 */
#define CHAR_SIGBITS(c, i) ((c) >> i & 1)
/**
 * @brief	Returns i-th bit in the bitmap pointed to by a char ptr 
 * 
 * @param	x	char * pointing to start of bitmap 
 * @param 	i	i-th bit 
 */
#define BITMAP_GET(x, i) ((*(x + (i / 8))) >> (i % 8) & 1)
/**
 * @brief 	Returns block address indexed by block number
 * 
 * @param	s	start address of disk volumne 
 * @param	i	block number
 */
#define BLOCK_ADDR(s, i) ((s) + ((i)*EXT2_BLOCK_SIZE))

/**
 * @brief   Converts inode number to inode index
 *      inode number:    1   2   3   4   5   ...
 *      inode index:     0   1   2   3   ...
 */
#define TO_INODE_INDEX(inode_number) ((inode_number)-1)

/**
 * @brief   Check if inode is occupied in bitmap 
 */
#define INODE_EXIST(s, i) (BITMAP_GET(s, (TO_INODE_INDEX(i))))
/**
 * @brief	Returns address of i-node struct given i-node number
 * 
 * 
 * @param	s	start address of inode table 
 * @param 	i	inode number
 * @param	n	size of i-node struct
 * 
 * @precondition    i >= 1
 */
#define INODE_ADDR(s, i, n) (s + (TO_INODE_INDEX(i)) * n)

/**
 * @brief   Prints bitmap 
 * 
 * @param addr  start address of the bitmap
 * @param size  number of bits in the bitmap
 *              32  for inode bitmap 
 *              128 for block bitmap
 */
void print_bitmap(unsigned char *addr, int size);

/**
 * @brief	Returns non-negative number i_mode is as specified 
 * 
 * @param	x	i_mode of i-node structure
 */
#define EXT2_ISREG(x) ((x & EXT2_S_IFREG) == EXT2_S_IFREG)  /* 1000 0000 0000 0000 */
#define EXT2_ISDIR(x) ((x & EXT2_S_IFDIR) == EXT2_S_IFDIR)  /* 0100 0000 0000 0000 */
#define EXT2_ISLINK(x) ((x & EXT2_S_IFLNK) == EXT2_S_IFLNK) /* 1010 0000 0000 0000 */

#define EXT2_INODE_ISREG(x) (EXT2_ISREG((x->i_mode)))
#define EXT2_INODE_ISDIR(x) (EXT2_ISDIR((x->i_mode)))
#define EXT2_INODE_ISLINK(x) (EXT2_ISLINK((x->i_mode)))

/**
 * @brief	Extract file type from inode 
 * 
 * @return  f   for regular file
 *          d   for directory
 *          l   for link 
 *          e   for error
 */
char inode_filetype(const struct ext2_inode *inode);

/**
 * @brief	Prints inode blocks occupied by inode
 */
void print_inode_blocks(const struct ext2_inode *inode);

/**
 * @brief   Returns block number for data block owned by an i-node
 */

/**
 * @brief	Prints selected fields in inodes 
 */
void print_inode(const struct ext2_inode *inode, int inode_number);

/**
 * @brief   Prints directory entries in inode, 
 * 
 * @precondition    inode is of type directory
 */
void print_directory_content(const struct ext2_inode *inode, int inode_number);

#endif