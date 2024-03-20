# Compute Imbalance ratio for each label
IRs = (float(train_oh_diagnoses_df.shape[0])
       / train_oh_diagnoses_df.sum(axis=0)) - 1.0
# IRs = IRs.clip(upper=25).to_numpy()
class_weights = np.ones(shape=IRs.shape + (2,))
class_weights[:, 1] = IRs

label_frequencies = train_oh_diagnoses_df.sum(axis=0)/float(train_oh_diagnoses_df.shape[0])

# Add the corresponding entry to the hyperparameters dict
hp_values['class_weight_cap'] = {'enable': True,
                                 'name': "IR_cap",
                                 'min_value': 1,
                                 'max_value': 200,
                                 'sampling': 'log',
                                 'default': 2,
                                 'label_frequencies': label_frequencies
                                 }
