#!/usr/bin/bash
aws-jupyter terminate
aws s3 rb s3://dsc291s3 --force
#TODO: generalize aws s3 rb call to all buckets