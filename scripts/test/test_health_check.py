from huggingface_hub import InferenceClient
client = InferenceClient("https://cr41uamktrsdyg3d.us-east-1.aws.endpoints.huggingface.cloud")
client.health_check()

