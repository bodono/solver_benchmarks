#!/bin/bash
cat miplib.tar.gz.* | tar xzvf -
mv miplib_data/* .
rm -rf miplib_data

