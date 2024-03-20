TARGETDIR="/home/pgdata/omop_csv_dump/"
CONCHAIN="dbname=mimic user=postgres password=XXXXXXX options=--search_path=omop"

psql "$CONCHAIN" -Atc "select tablename from pg_tables where schemaname='$SCHEMA'" |\
	  while read TBL; do
		      psql "$CONCHAIN" -c "COPY $SCHEMA.$TBL TO PROGRAM \'gzip > $TARGETDIR/$TBL.gz \' WITH CSV HEADER QUOTE '"'" 
	      done
