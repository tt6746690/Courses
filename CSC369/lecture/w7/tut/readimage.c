#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>

#include "ext2.h"
#include "utilities.h"

/* 
 *  only one blockgroup...
 */

int main(int argc, char **argv)
{

    if (argc != 2)
    {
        fprintf(stderr, "Usage: readimg <image file name>\n");
        exit(1);
    }
    int fd = open(argv[1], O_RDWR);

    disk = mmap(NULL, 128 * 1024, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (disk == MAP_FAILED)
    {
        perror("mmap");
        exit(1);
    }
    /* 
        Inodes: 32
        Blocks: 128
        Block group:
            block bitmap: 3
            inode bitmap: 4
            inode table: 5
            free blocks: 105
            free inodes: 21
            used_dirs: 2
    */

    struct ext2_super_block *sb = (struct ext2_super_block *)BLOCK_ADDR(disk, 1);
    struct ext2_group_desc *gd = (struct ext2_group_desc *)BLOCK_ADDR(disk, 2);
    unsigned char *ext2_block_bitmap = BLOCK_ADDR(disk, gd->bg_block_bitmap);
    unsigned char *ext2_inode_bitmap = BLOCK_ADDR(disk, gd->bg_inode_bitmap);
    unsigned char *ext2_inode_table = BLOCK_ADDR(disk, gd->bg_inode_table);

    /*
        Task 1 
    */
    printf("Inodes: %d\n", sb->s_inodes_count);
    printf("Blocks: %d\n", sb->s_blocks_count);
    printf("Block group:\n");
    printf("\tblock bitmap:%d\n", gd->bg_block_bitmap);
    printf("\tinode bitmap:%d\n", gd->bg_inode_bitmap);
    printf("\tinode table:%d\n", gd->bg_inode_table);
    printf("\tfree blocks: %d\n", gd->bg_free_blocks_count);
    printf("\tfree inodes: %d\n", gd->bg_free_inodes_count);
    printf("\tused dirs: %d\n", gd->bg_used_dirs_count);
    /*
        Task 2 
    */
    printf("Block bitmap: ");
    print_bitmap(ext2_block_bitmap, sb->s_blocks_count);
    printf("Inode bitmap: ");
    print_bitmap(ext2_inode_bitmap, sb->s_inodes_count);

    printf("Inodes:\n");

    int inode_size = sizeof(struct ext2_inode);
    int inode_number = EXT2_ROOT_INO;

    if (INODE_EXIST(ext2_inode_bitmap, inode_number))
    {
        print_inode((ino_ptr)INODE_ADDR(ext2_inode_table, inode_number, inode_size), inode_number);
    }

    for (inode_number = EXT2_FIRST_NONRESERVED_INO; inode_number <= INODE_TOTAL; ++inode_number)
    {
        if (INODE_EXIST(ext2_inode_bitmap, inode_number))
        {
            print_inode((ino_ptr)INODE_ADDR(ext2_inode_table, inode_number, inode_size), inode_number);
        }
    }
    /*
        Task 3: directory entry 

        Directory Blocks:
            DIR BLOCK NUM: 9 (for inode 2)
        Inode: 2 rec_len: 12 name_len: 1 type= d name=.
        Inode: 2 rec_len: 12 name_len: 2 type= d name=..
        Inode: 11 rec_len: 1000 name_len: 10 type= d name=lost+found
    */
    printf("Directory Blocks:\n");

    inode_number = EXT2_ROOT_INO;
    ino_ptr inode;

    if (INODE_EXIST(ext2_inode_bitmap, inode_number))
    {
        inode = (ino_ptr)INODE_ADDR(ext2_inode_table, inode_number, inode_size);
        print_directory_content(inode, inode_number);
    }

    for (inode_number = EXT2_FIRST_NONRESERVED_INO; inode_number <= INODE_TOTAL; ++inode_number)
    {
        inode = (ino_ptr)INODE_ADDR(ext2_inode_table, inode_number, inode_size);
        if (INODE_EXIST(ext2_inode_bitmap, inode_number) && EXT2_INODE_ISDIR(inode))
        {
            print_directory_content(inode, inode_number);
        }
    }

    if ((close(fd)) != 0)
    {
        perror("close");
        exit(0);
    }

    return 0;
}
