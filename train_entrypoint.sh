#!/bin/bash

# SageMaker Training Job Entrypoint Script
set -e

echo "🚀 Starting YunMin-Mamba training..."

# Check if we're running in SageMaker environment
if [[ -n "${SM_MODEL_DIR}" ]]; then
    echo "🔧 Running in SageMaker training environment"
    echo "📁 SM_CHANNEL_TRAINING: ${SM_CHANNEL_TRAINING:-not_set}"
    echo "📁 SM_MODEL_DIR: ${SM_MODEL_DIR}"
    echo "📁 SM_OUTPUT_DATA_DIR: ${SM_OUTPUT_DATA_DIR:-not_set}"
    
    # Create SageMaker directories if they don't exist
    mkdir -p "${SM_MODEL_DIR}"
    mkdir -p "${SM_OUTPUT_DATA_DIR:-/opt/ml/output}"
    
    # Set up logging for SageMaker
    exec python /app/train_mamba.py "$@" 2>&1 | tee "${SM_OUTPUT_DATA_DIR:-/opt/ml/output}/training.log"
else
    echo "🔧 Running in standalone mode"
    # Standalone execution
    exec python /app/train_mamba.py "$@"
fi 