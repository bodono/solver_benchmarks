#!/bin/bash
cat miplib.tar.gz.* | tar xzvf -
mv miplib/* .
rm -rf miplib

