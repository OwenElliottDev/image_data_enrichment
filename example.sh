./target/release/image_enrichment \
  --dir ./test_data \
  --api_url http://localhost:11434/api/chat \
  --model qwen3-vl:4b \
  --batch-size 2 \
  --schema example_schema.json \
  --pretty-json \
  --suffix "_processed" \
  --skip-existing