#include <stdio.h>
#include "utilities.h"

void print_bitmap(unsigned char *addr, int size)
{
    unsigned int idx = 1; /* inode starts from 1 */
    while (idx < size)
    {
        printf("%d", BITMAP_GET(addr, idx));
        if (++idx % 8 == 0)
            printf(" ");
    }
    printf("\n");
}

char inode_filetype(const struct ext2_inode *inode)
{
    const unsigned short imode = inode->i_mode; // 2 bytes

    if (EXT2_ISLINK(imode) != 0)
        return 'l';
    if (EXT2_ISREG(imode) != 0)
        return 'f';
    if (EXT2_ISDIR(imode) != 0)
        return 'd';
    return 'e';
}

void print_inode_blocks(const struct ext2_inode *inode)
{
    const unsigned int *blocks = inode->i_block;

    // direct pointer
    for (int i = 0; i < 12; i++)
    {
        if (!blocks[i])
        {
            break;
        }
        printf("%d  ", blocks[i]);
    }

    // singly indirect pointer
    if (blocks[12])
    {
        printf("%d ", blocks[12]);
        // go to data block and continue
    }
    printf("\n");
}

void print_inode(const struct ext2_inode *inode, int inode_number)
{
    printf("[%d] type: %c size: %d link: %d blocks: %d\n",
           inode_number, inode_filetype(inode), inode->i_size, inode->i_links_count, inode->i_blocks);
    printf("[%d] Blocks: ", inode_number);
    print_inode_blocks(inode);
}

void print_directory_content(const struct ext2_inode *inode, int inode_number)
{
    const unsigned int *blocks = inode->i_block;
    unsigned char *dir_entry_block;

    // direct pointer
    for (int i = 0; i < 12; i++)
    {
        if (!blocks[i])
        {
            break;
        }
        printf("\tDIR BLOCK NUM: %d (for inode %d)\n", blocks[i], inode_number);
        dir_entry_block = BLOCK_ADDR(disk, blocks[i]);

        /* rec_len for last entry fills the entire block */
        while (dir_entry_block != BLOCK_ADDR(disk, blocks[i] + 1))
        {
            dir_entry_ptr entry = (dir_entry_ptr)(dir_entry_block);
            entry->name[entry->name_len] = '\0';
            printf("Inode: %d rec_len: %d name_len: %d type= %c name=%s\n",
                   entry->inode, entry->rec_len, entry->name_len, entry->file_type, entry->name);
            dir_entry_block += entry->rec_len;
        }
    }

    // singly indirect pointer
    if (blocks[12])
    {
        printf("%d ", blocks[12]);
        // go to data block and continue
    }
}