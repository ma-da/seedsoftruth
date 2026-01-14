#!/bin/bash
curl -X POST http://localhost:5000/api/chat \
  -i \
  -b ./cookies.txt \
  -H "Content-Type: application/json" \
  -d '{
        "message": "Who killed JFK?",
        "user_id": "test_user",
        "force_queue": "true",
        "max_new_tokens": 500
      }'
