#!/bin/bash
curl -X POST http://localhost:5000/api/feedback \
  -i \
  -b ./cookies.txt \
  -H "Content-Type: application/json" \
  -d '{
        "job_id": "61d29d18-a2db-4d87-98d6-5016b7adb038",
        "relevance": 3,
        "accuracy": 8,
        "style": 10,
        "commends": "this was a good response"
      }'
