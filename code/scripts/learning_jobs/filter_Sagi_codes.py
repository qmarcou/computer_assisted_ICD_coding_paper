import project_miscellaneous as misc

if filter_Sagi:
    print("Filtering Sagi codes")
    drop_cols = misc.filter_Sagi_diagnoses(oh_diagnoses_df)

    train_oh_diagnoses_df = train_oh_diagnoses_df.drop(columns=drop_cols)
    dev_oh_diagnoses_df = dev_oh_diagnoses_df.drop(columns=drop_cols)
    test_oh_diagnoses_df = test_oh_diagnoses_df.drop(columns=drop_cols)


icd9_allcodes = icd9_allcodes[icd9_allcodes
    .isin(train_oh_diagnoses_df.columns)]
icd9_chapters = icd9_chapters[icd9_chapters
    .isin(train_oh_diagnoses_df.columns)]
icd9_subchapters = icd9_subchapters[icd9_subchapters
    .isin(train_oh_diagnoses_df.columns)]
icd9_3dig = icd9_3dig[icd9_3dig
    .isin(train_oh_diagnoses_df.columns)]
icd9_4dig = icd9_4dig[icd9_4dig
    .isin(train_oh_diagnoses_df.columns)]
icd9_5dig = icd9_5dig[icd9_5dig
    .isin(train_oh_diagnoses_df.columns)]
icd9_billing = icd9_billing[icd9_billing
    .isin(train_oh_diagnoses_df.columns)]