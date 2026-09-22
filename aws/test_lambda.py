from lambda_function import lambda_handler
import json

event = {
    "body": json.dumps({
        "step": 1,
        "transaction_type": "TRANSFER",
        "amount": 100000,
        "oldbalanceOrg": 100000,
        "newbalanceOrig": 0,
        "oldbalanceDest": 0,
        "newbalanceDest": 0,
        "model_choice": "Random Forest"
    })
}

print("\n========== TESTING LAMBDA ==========\n")

response = lambda_handler(event, None)

print("Lambda Response:")
print(json.dumps(response, indent=2))

print("\n====================================\n")