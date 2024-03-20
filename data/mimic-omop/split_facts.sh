# bin/bash
zcat fact_relationship.csv.gz | split --numeric-suffixes=1 -l 40000000 --filter='gzip > $FILE.gz' - fact_relationship.csv.part
