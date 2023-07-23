import bentoml
from bentoml.io import JSON

model_ref = bentoml.xgboost.get("credit_risk_pred_model:latest")
dv = model_ref.custom_objects["DictVectorizer"]

model_runner = model_ref.to_runner()

svc = bentoml.Service("credit_risk_classifier", runners= [model_runner])

@svc.api(input=JSON(), output=JSON())
def classify(application_data):
    vector = dv.transform(application_data)
    prediction = model_runner.predict.run(vector)
    print(prediction)
    return {"status": "Approved"}
    