#!/bin/bash

echo "checking existence of message file."
echo "Checking"

if [ -f /var/log/messages ]
then 
    echo "/var/log/messages exist"
fi

echo 
echo "...done"
