# helper functions
def set_model_params(model, param_map):
    for param, value in param_map.items():
        model.set(param, value)
    return model

# general models training process
def basic_model_train(train_set, test_set, model, evaluator,
                      paramGrid, metrics=['rmse', 'mae', 'r2']):
    min_score = float('inf') # very large float
    best_model = None
    
    for params in paramGrid:
        # set parameters
        model  = set_model_params(model, params)
        
        # train the model
        trained_model = model.fit(train_set)
        
        # evaluate on trainset
        preds_train = trained_model.transform(train_set)
        score = evaluator.evaluate(preds_train)

        # check if current model is the best so far
        if score < min_score:
            best_model = trained_model
            min_score = score
            
    preds_test = best_model.transform(test_set)
    train_metrics = {}
    test_metrics = {}
    for m in metrics:
        test_metrics[m]= evaluator.setMetricName(m).evaluate(preds_test)
        train_metrics[m]= evaluator.setMetricName(m).evaluate(preds_train)
            
    return best_model, train_metrics, test_metrics
