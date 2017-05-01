#!/bin/bash
for f in *
do 
    if [ ! -d "$f" ]
    then 

        echo $f ":" `du -k "$f" | cut -d " " -f 1`
    fi
done
    
