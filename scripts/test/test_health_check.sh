
HEALTH_CHECK_URL=https://cr41uamktrsdyg3d.us-east-1.aws.endpoints.huggingface.cloud/health

curl $HEALTH_CHECK_URL \
     -H "Authorization: Bearer $HF_TOKEN" \
     -d '{"inputs": "Hello world"}'

