#!/bin/bash

if [[ $(uname) == "Linux" ]]; then
    if command -v yum &> /dev/null; then
        yum update -y && yum install -y glibc-devel wget python3-devel
    elif command -v apt-get &> /dev/null; then
        apt-get update && apt-get install -y libc6-dev wget python3-dev
    elif command -v apk &> /dev/null; then
        apk add --update wget python3-dev
    else
        echo "Unsupported package manager"
        exit 1
    fi
else
    echo "This script is intended only for Linux"
    exit 1
fi
