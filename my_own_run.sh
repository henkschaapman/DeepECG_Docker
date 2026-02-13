# 1. Create dummy API key (required but won't be used)
echo '{"HUGGING_FACE_API_KEY": "dummy"}' > /tmp/dummy_api_key.json

# 3. Build Docker image (if not already built)
cd DeepECG_Docker
docker build -t deepecg-docker .

# 4. Run with GPU
docker run --gpus "device=0" \
  -v ${CSV_DIR}:/app/inputs:ro \
  -v ${OUTPUT_DIR}:/app/outputs \
  -v ${ECG_SIGNALS}:/app/ecg_signals:ro \
  -v ${MODEL_DIR}:/app/weights/wcr_afib_5y:ro \
  -v $(pwd)/preprocessing:/app/preprocessing \
  -v /tmp/dummy_api_key.json:/app/api_key.json:ro \
  deepecg-docker \
  bash run_pipeline.bash --mode full_run --csv_file_name ${CSV_FILE}

# 5. Check results
ls -lh ${OUTPUT_DIR}/