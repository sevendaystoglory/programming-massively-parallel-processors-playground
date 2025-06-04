#!/bin/bash

git add . -v
if [ -z "$1" ]; then
    comment="autocomment:$RANDOM$RANDOM" 
    git commit -m "$comment"
    echo "pushing with comment: $comment" 
else
    git commit -m "$1" 
fi 
git push origin main