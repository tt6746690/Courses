RSYNC = /usr/local/Cellar/rsync/3.1.3_1/bin/rsync
RSYNCTAGS = --archive --verbose --info=progress2 -au

HOST_FOLDER = $(HOME)/School/c_2018_2019/CSC320
# REMOTE = wpq@$(COMPS0)
# REMOTE_FOLDER = /u/wpq/github/CSC320
REMOTE = wangpeiq@$(CDF)
REMOTE_FOLDER = /h/u13/c6/00/wangpeiq/github/CSC320

sync:
	$(RSYNC) $(RSYNCTAGS) $(HOST_FOLDER)/ $(REMOTE):$(REMOTE_FOLDER)
sync_rev:
	$(RSYNC) $(RSYNCTAGS) $(REMOTE):$(REMOTE_FOLDER)/ $(HOST_FOLDER)

vscode:
	sshcode --skipsync $(REMOTE) $(REMOTE_FOLDER)