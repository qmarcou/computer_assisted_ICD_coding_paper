# /bin/#!/usr/bin/env bash -e
zcat fact_relationship.csv.part0* | gzip > fact_relationship.csv.gz
