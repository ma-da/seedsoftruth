HEALTH_CHECK_URL=https://cr41uamktrsdyg3d.us-east-1.aws.endpoints.huggingface.cloud

curl $HEALTH_CHECK_URL \
  -X POST \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"inputs": "health_check"}'
