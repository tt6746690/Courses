RSYNC = /usr/local/Cellar/rsync/3.1.3_1/bin/rsync
RSYNCTAGS = --archive --verbose --info=progress2 -au

SRC_FOLDER = $(HOME)/School/c_2018_2019/CSC367/labs

REMOTE = wpq@$(COMPS0)
LAB_FOLDER = /u/wpq/github/CSC367/labs

sync:
	$(RSYNC) $(RSYNCTAGS) $(SRC_FOLDER)/ $(REMOTE):$(LAB_FOLDER)
sync_rev:
	$(RSYNC) $(RSYNCTAGS) $(REMOTE):$(LAB_FOLDER)/ $(SRC_FOLDER)


LOCAL_ASSIGNMENTS = $(HOME)/School/c_2018_2019/CSC367/assignments
SCINET_REMOTE = csc367student028@$(SCINET)
SCINET_ASSIGNMENTS = /home/t/teachcsc367/csc367student028/.me/assignments

ssync:
	$(RSYNC) $(RSYNCTAGS) $(LOCAL_ASSIGNMENTS)/ $(SCINET_REMOTE):$(SCINET_ASSIGNMENTS)
ssync_rev:
	$(RSYNC) $(RSYNCTAGS) $(SCINET_REMOTE):$(SCINET_ASSIGNMENTS)/ $(LOCAL_ASSIGNMENTS)
