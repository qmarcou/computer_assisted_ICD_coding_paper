import pandas as pd

tuner.save()
n_models = 5
best_models = tuner.get_best_models(num_models=n_models)
best_hps = tuner.get_best_hyperparameters(num_trials=n_models)
for model, hp, count in zip(best_models, best_hps, range(1, n_models + 1)):
    dir = tuner.project_dir + "/best_models/" + str(count)+"/"
    model.save(dir + "/model")
    # Save current model hyperparameters to file
    with open(dir+'model_hyperparameters.json', "w") as outfile:
        json.dump(hp.values, outfile)

    # Evaluate the model on the 3 datasets
    try:
        # Recompile the model to have all the desired metrics
        model.compile(metrics=evaluation_metrics)
        eval_train = model.evaluate(
            x=train_oh_ingredients_df, y=train_oh_diagnoses_df,
            return_dict=True, **evaluate_kwargs)
        eval_dev = model.evaluate(
            x=dev_oh_ingredients_df, y=dev_oh_diagnoses_df,
            return_dict=True, **evaluate_kwargs)
        eval_test = model.evaluate(
            x=test_oh_ingredients_df, y=test_oh_diagnoses_df,
            return_dict=True, **evaluate_kwargs)

    except Exception as e:
        print("Exception caught during evaluation of the best models")
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)
    else:
        with open(dir+"evaluate_train.json", "w") as outfile:
            json.dump(eval_train, outfile)
        with open(dir + "evaluate_dev.json", "w") as outfile:
            json.dump(eval_dev, outfile)
        with open(dir+"evaluate_test.json", "w") as outfile:
            json.dump(eval_test, outfile)

    try:
        y_pred = model.predict(
            x=test_oh_ingredients_df,
            **predict_kwargs)

    except Exception as e:
        print("Exception caught during prediction with the best models")
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)
    else:
        try:
            y_pred = pd.DataFrame(y_pred, columns=test_oh_diagnoses_df.columns,
                                  index=test_oh_diagnoses_df.index)
            y_pred.to_csv(dir+"predictions_test.csv.gz",
                          header=True,
                          index=True,
                          index_label="visit_occurrence_id")
        except Exception as e:
            print("Exception caught writing best model predictions to file")
            if hasattr(e, 'message'):
                print(e.message)
            else:
                print(e)
