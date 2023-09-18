#!/bin/sh

echo $1 > VERSION && 
    sed -i "s/\"version\": \".*\"/\"version\": \"$1\"/" package.json
