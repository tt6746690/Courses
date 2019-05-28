RSYNC = /usr/local/Cellar/rsync/3.1.3_1/bin/rsync
RSYNCTAGS = --archive --verbose --info=progress2 -au

SRC_FOLDER = $(HOME)/School/c_2018_2019/CSC367/labs
REMOTE = wpq@$(COMPS0)
REMOTE_FOLDER = /u/wpq/github/CSC367/labs

sync:
	$(RSYNC) $(RSYNCTAGS) $(SRC_FOLDER)/ $(REMOTE):$(REMOTE_FOLDER)
sync_rev:
	$(RSYNC) $(RSYNCTAGS) $(REMOTE):$(REMOTE_FOLDER)/ $(SRC_FOLDER)
vscode:
	sshcode --skipsync wpq@comps0.cs.toronto.edu /h/96/wpq/github/

LOCAL_ASSIGNMENTS = $(HOME)/School/c_2018_2019/CSC367/assignments
LOCAL_LABS = $(HOME)/School/c_2018_2019/CSC367/labs
SCINET_REMOTE = csc367student028@$(SCINET)
SCINET_ASSIGNMENTS = /home/t/teachcsc367/csc367student028/.me/assignments
SCINET_LABS = /home/t/teachcsc367/csc367student028/.me/labs

ssync:
	# $(RSYNC) $(RSYNCTAGS) $(LOCAL_ASSIGNMENTS)/ $(SCINET_REMOTE):$(SCINET_ASSIGNMENTS)
	$(RSYNC) $(RSYNCTAGS) $(LOCAL_LABS)/ $(SCINET_REMOTE):$(SCINET_LABS)
ssync_rev:
	# $(RSYNC) $(RSYNCTAGS) $(SCINET_REMOTE):$(SCINET_ASSIGNMENTS)/ $(LOCAL_ASSIGNMENTS)
	$(RSYNC) $(RSYNCTAGS) $(SCINET_REMOTE):$(SCINET_LABS)/ $(LOCAL_LABS)
