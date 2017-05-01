


_Devices_

character devices
+ unbuffered, direct access to hardware service
+ usually does not allow read/write

block devices
+ provide buffered access to hardware devices
+ allow read/write of any length


_Inode_

The inode is a data structure in a Unix-style file system which describes a filesystem object such as a file or a directory. Each inode stores the attributes and disk block location(s) of the object's data. Filesystem object attributes may include metadata (times of last change, access, modification), as well as owner and permission data.


`stat`
+ displays information about the file pointed to by file
