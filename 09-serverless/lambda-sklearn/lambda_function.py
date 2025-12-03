import pickle

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)


def lambda_handler(event, context):
    customer = event['customer']
    prob = predict_single(customer)
    return {
        'statusCode': 200,
        'body': {
            'churn_prob': prob,
            'churn_decision': bool(prob >= 0.5)
        }
    }