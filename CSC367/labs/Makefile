RSYNC = /usr/local/Cellar/rsync/3.1.3_1/bin/rsync
RSYNCTAGS = --archive --verbose --info=progress2 -au

SRC_FOLDER = $(HOME)/School/c_2018_2019/CSC367/labs

REMOTE = wpq@$(COMPS0)
DST_FOLDER = /u/wpq/github/CSC367/labs

sync:
	$(RSYNC) $(RSYNCTAGS) $(SRC_FOLDER)/ $(REMOTE):$(DST_FOLDER)
revsync:
	$(RSYNC) $(RSYNCTAGS) $(REMOTE):$(DST_FOLDER)/ $(SRC_FOLDER)
