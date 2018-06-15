# Big Data ML Pipeline

## Overview
Production-ready distributed machine learning pipeline using Apache Spark MLlib for training, evaluating, and deploying ML models at scale. Supports multiple algorithms, automated feature engineering, hyperparameter tuning, and model versioning.

## Technologies
- Apache Spark 3.5+
- PySpark MLlib
- Python 3.9+
- MLflow 2.8+
- Delta Lake 3.0+
- FastAPI
- Docker & Kubernetes

## Features
- **Distributed Training**: Train models on massive datasets using Spark's distributed computing
- **Feature Engineering**: Automated feature extraction, transformation, and selection at scale
- **Model Evaluation**: Comprehensive metrics, cross-validation, and performance tracking
- **Hyperparameter Tuning**: Grid search and random search with distributed execution
- **Pipeline Persistence**: Save and load complete ML pipelines
- **Model Registry**: MLflow integration for model versioning and tracking
- **Real-time Scoring**: REST API for batch and real-time predictions
- **A/B Testing**: Support for model comparison and champion/challenger patterns

## Architecture
```
Data Sources → Feature Engineering → Model Training → Evaluation → Deployment
     ↓              (Spark)              (MLlib)        (MLflow)      (API)
  Delta Lake    → Transformers →    Algorithms    →   Metrics   →  Serving
```

## Project Structure
```
big-data-ml-pipeline/
├── src/
│   ├── ml_pipeline.py          # Main ML pipeline orchestration
│   ├── feature_engineering.py  # Feature transformers
│   ├── model_trainer.py        # Model training logic
│   └── model_evaluator.py      # Evaluation metrics
├── api/
│   └── prediction_api.py       # FastAPI serving endpoint
├── config/
│   └── pipeline_config.yaml    # Pipeline configuration
├── tests/
│   └── test_pipeline.py        # Unit tests
├── notebooks/
│   └── model_exploration.ipynb # Jupyter notebook
├── terraform/
│   └── main.tf                 # Infrastructure
└── docker-compose.yml          # Local development

```

## Supported Algorithms
- **Classification**: Logistic Regression, Random Forest, GBT, Naive Bayes
- **Regression**: Linear Regression, Random Forest, GBT, GLM
- **Clustering**: K-Means, Bisecting K-Means, GMM
- **Recommendation**: ALS (Alternating Least Squares)

## Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start Spark cluster
docker-compose up -d

# Run training pipeline
python src/ml_pipeline.py --config config/pipeline_config.yaml

# Start API server
python api/prediction_api.py
```

### API Usage
```bash
# Health check
curl http://localhost:8000/health

# Train model
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "random_forest", "data_path": "s3://bucket/data"}'

# Make predictions
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "rf_v1", "features": [1.2, 3.4, 5.6]}'
```

## Configuration
Edit `config/pipeline_config.yaml`:
```yaml
spark:
  app_name: "ML Pipeline"
  master: "spark://localhost:7077"
  
data:
  input_path: "s3://data-lake/raw/"
  output_path: "s3://data-lake/models/"
  
model:
  algorithm: "random_forest"
  hyperparameters:
    numTrees: 100
    maxDepth: 10
```

## Deployment

### Kubernetes
```bash
kubectl apply -f k8s/deployment.yaml
```

### AWS EMR
```bash
aws emr create-cluster --name "ML Pipeline" \
  --release-label emr-6.15.0 \
  --applications Name=Spark \
  --instance-type m5.xlarge \
  --instance-count 3
```

## Monitoring
- MLflow UI: http://localhost:5000
- Spark UI: http://localhost:4040
- API Metrics: http://localhost:8000/metrics

## Performance
- Trains on 1TB+ datasets
- Supports 10,000+ features
- Sub-second prediction latency
- Horizontal scaling with Spark

## License
MIT License
