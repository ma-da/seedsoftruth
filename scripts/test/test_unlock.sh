#!/bin/bash
curl -X POST http://localhost:5000/api/unlock \
  -i \
  -c ./cookies.txt \
  -H "Content-Type: application/json" \
  -d '{
        "password": "irate_kittens"
      }'
