#!/bin/bash
curl -X POST http://localhost:5000/api/status \
  -H "Content-Type: application/json" \
  -d '{
      }'
