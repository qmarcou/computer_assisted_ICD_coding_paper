filename = "/tmp/icd9chapters_concepts.csv"
write("concept_id,concept_name,domain_id,vocabulary_id,concept_class_id,standard_concept,concept_code",
      file=filename,
      ncolumns = 1, append = FALSE, sep = "")
i = 201000000
for (name in names(icd9Chapters)) {
  write(paste(format(i, scientific=F),paste0('"',name,'"'),
              "Condition","ICD9CM","Chapter","",
              paste0(icd9Chapters[[name]]['start'],"-",icd9Chapters[[name]]['end']),
              sep = ","),
        file = filename, ncolumns = 1, append = TRUE, sep = "")
  i=i+1
}

for (name in names(icd9ChaptersSub)) {
  write(paste(format(i, scientific=F),paste0('"',name,'"'),
              "Condition","ICD9CM","SubChapter","",
              paste0(icd9ChaptersSub[[name]]['start'],"-",icd9ChaptersSub[[name]]['end']),
              sep = ","),
        file = filename, ncolumns = 1, append = TRUE, sep = "")
  i=i+1
}
