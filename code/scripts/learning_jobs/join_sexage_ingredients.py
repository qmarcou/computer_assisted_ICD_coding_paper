# Join ingredients with age/sex information for input
train_oh_ingredients_df = pd.merge(age_sex_df, train_oh_ingredients_df,
                                   left_index=True,
                                   right_index=True,
                                   how='right')

dev_oh_ingredients_df = pd.merge(age_sex_df, dev_oh_ingredients_df,
                                 left_index=True,
                                 right_index=True,
                                 how='right')

test_oh_ingredients_df = pd.merge(age_sex_df, test_oh_ingredients_df,
                                 left_index=True,
                                 right_index=True,
                                 how='right')