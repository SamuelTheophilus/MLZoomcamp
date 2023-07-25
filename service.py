import logging

import bentoml
from bentoml.io import JSON

model_ref = bentoml.xgboost.get("credit_risk_pred_model:latest")
dv = model_ref.custom_objects["DictVectorizer"]

model_runner = model_ref.to_runner()
# Instantiate a service for the model_runner
svc = bentoml.Service("credit_risk_classifier", runners=[model_runner])

@svc.api(input=JSON(), output=JSON())
def classify(application_data):
    try:
        # List of all required features
        required_features = ['age', 'amount', 'assets', 'debt', 'expenses', 'home=ignore',
                             'home=other', 'home=owner', 'home=parents', 'home=private',
                             'home=rent', 'home=unk', 'income', 'job=fixed', 'job=freelance',
                             'job=others', 'job=partime', 'job=unk', 'marital=divorced',
                             'marital=married', 'marital=separated', 'marital=single',
                             'marital=widow', 'price', 'records=no', 'records=yes', 'seniority',
                             'time']
        
        # Check if each feature is in application_data and if not, add it with a default value
        for feature in required_features:
            if feature not in application_data:
                # If the feature is categorical (contains '='), use 'unknown' as the default value
                if '=' in feature:
                    application_data[feature] = 'unknown'
                # Otherwise, it's numeric, so use 0 as the default value
                else:
                    application_data[feature] = 0

        # Continue with the original prediction code
        vector = dv.transform(application_data)
        logging.info(f"This is the vector {vector}")
        prediction = model_runner.predict.run(vector)
        logging.info(prediction)
        status = "Approved" if prediction == 1 else "Rejected"
        return {"status": status}
    except Exception as e:
        logging.error("Error occurred: ", e)
        return {"status": "Error occurred during prediction"}