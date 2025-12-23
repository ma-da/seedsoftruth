#!/bin/bash
curl -X POST http://localhost:5000/api/chat \
  -i \
  -b ./cookies.txt \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "Who killed JFK?",
        "max_new_tokens": 500
      }'
