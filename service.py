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
        vector = dv.transform(application_data)
        logging.info(f"This is the vector {vector}")
        prediction = model_runner.predict.run(vector)
        logging.info(prediction)
        # Determine the status based on the prediction
        status = "Approved" if prediction == 1 else "Rejected"
        return {"status": status}
    except Exception as e:
        logging.error("Error occurred: ", e)
        return {"status": "Error occurred during prediction"}
