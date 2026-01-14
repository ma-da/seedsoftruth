#!/bin/bash
curl -X POST http://localhost:5000/api/queue \
  -i \
  -b ./cookies.txt \
  -H "Content-Type: application/json" \
  -d '{
        "user_id": "test_user"
      }'
