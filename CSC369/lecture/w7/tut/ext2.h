/* MODIFIED by Karen Reid for CSC369
 * to remove some of the unnecessary components */

/* MODIFIED by Tian Ze Chen for CSC369
 * to clean up the code and fix some bugs */

/*
 * Copyright (C) 1992, 1993, 1994, 1995
 * Remy Card (card@masi.ibp.fr)
 * Laboratoire MASI - Institut Blaise Pascal
 * Universite Pierre et Marie Curie (Paris VI)
 *
 *  from
 *
 *  linux/include/linux/minix_fs.h
 *
 *  Copyright (C) 1991, 1992  Linus Torvalds
 */

/*
 * invalide = inode 1
 * root = inode 2 
 */

#ifndef CSC369A3_EXT2_FS_H
#define CSC369A3_EXT2_FS_H

#define EXT2_BLOCK_SIZE 1024

/*
 * Structure of the super block: totals 1024bytes for one struct
 */
struct ext2_super_block
{
	unsigned int s_inodes_count;   /* Inodes count */
	unsigned int s_blocks_count;   /* Blocks count */
	unsigned int s_r_blocks_count; /* Reserved blocks count */

	unsigned int s_free_blocks_count; /* Free blocks count */
	unsigned int s_free_inodes_count; /* Free inodes count */
	unsigned int s_first_data_block;  /* First Data Block */
	unsigned int s_log_block_size;	/* Block size: in powers of 2 in kb, so if 0 the 2^0 = 1 kb block size */
	unsigned int s_log_frag_size;	 /* Fragment size */
	unsigned int s_blocks_per_group;  /* # Blocks per group */
	unsigned int s_frags_per_group;   /* # Fragments per group */
	unsigned int s_inodes_per_group;  /* # Inodes per group */

	unsigned int s_mtime;			/* Mount time */
	unsigned int s_wtime;			/* Write time */
	unsigned short s_mnt_count;		/* Mount count */
	unsigned short s_max_mnt_count; /* Maximal mount count */

	unsigned short s_magic;			  /* Magic signature: indicates ext2 filesystem */
	unsigned short s_state;			  /* File system state */
	unsigned short s_errors;		  /* Behaviour when detecting errors */
	unsigned short s_minor_rev_level; /* minor revision level */
	unsigned int s_lastcheck;		  /* time of last check */
	unsigned int s_checkinterval;	 /* max. time between checks */
	unsigned int s_creator_os;		  /* OS */
	unsigned int s_rev_level;		  /* Revision level */
	unsigned short s_def_resuid;	  /* Default uid for reserved blocks */
	unsigned short s_def_resgid;	  /* Default gid for reserved blocks */
	/*
	 * These fields are for EXT2_DYNAMIC_REV superblocks only.
	 *
	 * Note: the difference between the compatible feature set and
	 * the incompatible feature set is that if there is a bit set
	 * in the incompatible feature set that the kernel doesn't
	 * know about, it should refuse to mount the filesystem.
	 *
	 * e2fsck's requirements are more strict; if it doesn't know
	 * about a feature in either the compatible or incompatible
	 * feature set, it must abort and not try to meddle with
	 * things it doesn't understand...
	 */
	unsigned int s_first_ino;			   /* First non-reserved inode */
	unsigned short s_inode_size;		   /* size of inode structure */
	unsigned short s_block_group_nr;	   /* block group # of this superblock */
	unsigned int s_feature_compat;		   /* compatible feature set */
	unsigned int s_feature_incompat;	   /* incompatible feature set */
	unsigned int s_feature_ro_compat;	  /* readonly-compatible feature set */
	unsigned char s_uuid[16];			   /* 128-bit uuid for volume */
	char s_volume_name[16];				   /* volume name */
	char s_last_mounted[64];			   /* directory where last mounted */
	unsigned int s_algorithm_usage_bitmap; /* For compression */
	/*
	 * Performance hints.  Directory preallocation should only
	 * happen if the EXT2_COMPAT_PREALLOC flag is on.
	 */
	unsigned char s_prealloc_blocks;	 /* Nr of blocks to try to preallocate*/
	unsigned char s_prealloc_dir_blocks; /* Nr to preallocate for dirs */
	unsigned short s_padding1;
	/*
	 * Journaling support valid if EXT3_FEATURE_COMPAT_HAS_JOURNAL set.
	 */
	unsigned char s_journal_uuid[16]; /* uuid of journal superblock */
	unsigned int s_journal_inum;	  /* inode number of journal file */
	unsigned int s_journal_dev;		  /* device number of journal file */
	unsigned int s_last_orphan;		  /* start of list of inodes to delete */
	unsigned int s_hash_seed[4];	  /* HTREE hash seed */
	unsigned char s_def_hash_version; /* Default hash version to use */
	unsigned char s_reserved_char_pad;
	unsigned short s_reserved_word_pad;
	unsigned int s_default_mount_opts;
	unsigned int s_first_meta_bg; /* First metablock block group */
	unsigned int s_reserved[190]; /* Padding to the end of the block */
};

/*
 * Structure of a blocks group descriptor
 */
struct ext2_group_desc
{
	unsigned int bg_block_bitmap; /* Blocks bitmap block, as absolute reference to disk. 
											the location indicated by this field points to a block of size say 1kb (say), 
											equates to 8192 bits, so maximum 8192 possible blocks to track usage within a block group */

	unsigned int bg_inode_bitmap; /* Inodes bitmap block */

	unsigned int bg_inode_table;		 /* Inodes table block, 
											1st inode -> invalid file
											2nd inode -> root 
											... 
										*/
	unsigned short bg_free_blocks_count; /* Free blocks count */
	unsigned short bg_free_inodes_count; /* Free inodes count */
	unsigned short bg_used_dirs_count;   /* Directories count */
	unsigned short bg_pad;
	unsigned int bg_reserved[3];
};

/*
 * Structure of an inode on the disk
 */

struct ext2_inode
{
	/* metadata */
	unsigned short i_mode;		  /* File mode, or file type, dir, link, regular file...*/
	unsigned short i_uid;		  /* Low 16 bits of Owner Uid */
	unsigned int i_size;		  /* Size in bytes */
	unsigned int i_atime;		  /* Access time */
	unsigned int i_ctime;		  /* Creation time */
	unsigned int i_mtime;		  /* Modification time */
	unsigned int i_dtime;		  /* Deletion Time */
	unsigned short i_gid;		  /* Low 16 bits of Group Id */
	unsigned short i_links_count; /* Links count */
	unsigned int i_blocks;		  /* Blocks count IN DISK SECTORS*/
	unsigned int i_flags;		  /* File flags */
	unsigned int osd1;			  /* OS dependent 1 */

	unsigned int i_block[15]; /*    Pointers to blocks: total of 15 pointers... 
									-- 1~12 direct 
									-- 13 singly indirect, points to a block (assume 1KB), containing 128 8bytes pointers
									-- 14 doubly indirect 
									-- 15 triply indirect
									if i_mode is 
									-- file, block 1 contains file 
									-- directory, block 1 contains a list of ext2_dir_entry 
								*/

	unsigned int i_generation; /* File version (for NFS) */
	unsigned int i_file_acl;   /* File ACL */
	unsigned int i_dir_acl;	/* Directory ACL */
	unsigned int i_faddr;	  /* Fragment address */
	unsigned int extra[3];
};

/*
 * Type field for file mode
 */

/* #define EXT2_S_IFSOCK 0xC000 */ /* socket */
#define EXT2_S_IFLNK 0xA000		   /* symbolic link */
#define EXT2_S_IFREG 0x8000		   /* regular file */
/* #define EXT2_S_IFBLK  0x6000 */ /* block device */
#define EXT2_S_IFDIR 0x4000		   /* directory */
/* #define EXT2_S_IFCHR  0x2000 */ /* character device */
/* #define EXT2_S_IFIFO  0x1000 */ /* fifo */

/*
 * Special inode numbers
 */

/* #define EXT2_BAD_INO          1 */ /* Bad blocks inode */
#define EXT2_ROOT_INO 2				  /* Root inode */
/* #define EXT4_USR_QUOTA_INO    3 */ /* User quota inode */
/* #define EXT4_GRP_QUOTA_INO    4 */ /* Group quota inode */
/* #define EXT2_BOOT_LOADER_INO  5 */ /* Boot loader inode */
/* #define EXT2_UNDEL_DIR_INO    6 */ /* Undelete directory inode */
/* #define EXT2_RESIZE_INO       7 */ /* Reserved group descriptors inode */
/* #define EXT2_JOURNAL_INO      8 */ /* Journal inode */
/* #define EXT2_EXCLUDE_INO      9 */ /* The "exclude" inode, for snapshots */
/* #define EXT4_REPLICA_INO     10 */ /* Used by non-upstream feature */

/* First non-reserved inode for old ext2 filesystems */
#define EXT2_GOOD_OLD_FIRST_INO 11

/*
 * Structure of a directory entry
 */

#define EXT2_NAME_LEN 255

/* WARNING: DO NOT use this struct, ext2_dir_entry_2 is the
 * one to use for the assignement */
struct ext2_dir_entry
{
	unsigned int inode;		 /* Inode number */
	unsigned short rec_len;  /* Directory entry length */
	unsigned short name_len; /* Name length */
	char name[];			 /* File name, up to EXT2_NAME_LEN */
};

/*
 * The new version of the directory entry.  Since EXT2 structures are
 * stored in intel byte order, and the name_len field could never be
 * bigger than 255 chars, it's safe to reclaim the extra byte for the
 * file_type field.
 */

struct ext2_dir_entry_2
{
	/* size
		first 4: 8 bytes
	*/
	unsigned int inode;		/* Inode number */
	unsigned short rec_len; /* Directory entry length, always power of 2,  */
	unsigned char name_len; /* Name length, actual length of string,  may not be power of 2*/
	unsigned char file_type;
	char name[]; /* File name, up to EXT2_NAME_LEN=255 */
};

/*
 * Ext2 directory file types.  Only the low 3 bits are used.  The
 * other bits are reserved for now.
 */

#define EXT2_FT_UNKNOWN 0		 /* Unknown File Type */
#define EXT2_FT_REG_FILE 1		 /* Regular File */
#define EXT2_FT_DIR 2			 /* Directory File */
/* #define EXT2_FT_CHRDEV   3 */ /* Character Device */
/* #define EXT2_FT_BLKDEV   4 */ /* Block Device */
/* #define EXT2_FT_FIFO     5 */ /* Buffer File */
/* #define EXT2_FT_SOCK     6 */ /* Socket File */
#define EXT2_FT_SYMLINK 7		 /* Symbolic Link */

#define EXT2_FT_MAX 8

#endif
