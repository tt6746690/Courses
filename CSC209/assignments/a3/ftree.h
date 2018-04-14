#ifndef _FTREE_H_
#define _FTREE_H_

/* Function for copying a file tree rooted at src to dest
 * Returns < 0 on error. The magnitude of the return value
 * is the number of processes involved in the copy and is
 * at least 1.
 */
int copy_ftree(const char *src, const char *dest);

#endif // _FTREE_H_
