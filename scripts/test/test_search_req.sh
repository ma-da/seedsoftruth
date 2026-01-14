#!/bin/bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{
        "query": "What do declassified documents say about JFK assassination planning?",
        "max_n": 10
      }'
