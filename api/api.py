"""
ML Pipeline Prediction API
FastAPI service for model training and predictions
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import uvicorn
import mlflow
from pyspark.sql import SparkSession
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ml_pipeline import MLPipelineOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Big Data ML Pipeline API",
    description="Distributed ML training and prediction service",
    version="1.0.0"
)

# Global state
pipeline_orchestrator = None
training_jobs = {}


class TrainingRequest(BaseModel):
    """Training request model"""
    data_path: str = Field(..., description="Path to training data")
    algorithm: str = Field("random_forest", description="ML algorithm to use")
    task_type: str = Field("classification", description="Task type: classification, regression, clustering")
    hyperparameter_tuning: bool = Field(False, description="Enable hyperparameter tuning")
    config_overrides: Optional[Dict[str, Any]] = Field(None, description="Configuration overrides")


class PredictionRequest(BaseModel):
    """Prediction request model"""
    model_path: str = Field(..., description="Path to trained model")
    features: List[List[float]] = Field(..., description="Feature vectors for prediction")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model"""
    model_path: str = Field(..., description="Path to trained model")
    data_path: str = Field(..., description="Path to data for prediction")
    output_path: str = Field(..., description="Path to save predictions")


class ModelInfo(BaseModel):
    """Model information"""
    model_id: str
    algorithm: str
    task_type: str
    metrics: Dict[str, float]
    created_at: str


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    global pipeline_orchestrator
    logger.info("Starting ML Pipeline API...")
    
    # Initialize pipeline orchestrator
    config_path = os.getenv("CONFIG_PATH", "config/pipeline_config.yaml")
    if os.path.exists(config_path):
        pipeline_orchestrator = MLPipelineOrchestrator(config_path)
        logger.info("Pipeline orchestrator initialized")
    else:
        logger.warning(f"Config file not found: {config_path}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global pipeline_orchestrator
    if pipeline_orchestrator:
        pipeline_orchestrator.stop()
    logger.info("ML Pipeline API shutdown complete")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Big Data ML Pipeline API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "pipeline_initialized": pipeline_orchestrator is not None
    }


@app.post("/train", response_model=Dict[str, Any])
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train a new ML model"""
    if not pipeline_orchestrator:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    job_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        logger.info(f"Starting training job {job_id}")
        
        # Load data
        df = pipeline_orchestrator.load_data(request.data_path)
        
        # Train model
        if request.hyperparameter_tuning:
            model = pipeline_orchestrator.hyperparameter_tuning(
                df, request.algorithm, request.task_type
            )
        else:
            model = pipeline_orchestrator.train(
                df, request.algorithm, request.task_type
            )
        
        # Save model
        model_path = f"models/{job_id}"
        pipeline_orchestrator.save_model(model_path)
        
        training_jobs[job_id] = {
            "status": "completed",
            "algorithm": request.algorithm,
            "task_type": request.task_type,
            "model_path": model_path,
            "completed_at": datetime.now().isoformat()
        }
        
        return {
            "job_id": job_id,
            "status": "completed",
            "model_path": model_path,
            "message": "Model training completed successfully"
        }
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        training_jobs[job_id] = {
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make predictions using trained model"""
    if not pipeline_orchestrator:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        # Load model
        pipeline_orchestrator.load_model(request.model_path)
        
        # Create DataFrame from features
        spark = pipeline_orchestrator.spark
        schema = ["feature_" + str(i) for i in range(len(request.features[0]))]
        df = spark.createDataFrame(request.features, schema=schema)
        
        # Make predictions
        predictions = pipeline_orchestrator.predict(df)
        
        # Convert to list
        results = predictions.select("prediction").collect()
        predictions_list = [row.prediction for row in results]
        
        return {
            "predictions": predictions_list,
            "count": len(predictions_list)
        }
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """Make batch predictions on large dataset"""
    if not pipeline_orchestrator:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        # Load model
        pipeline_orchestrator.load_model(request.model_path)
        
        # Load data
        df = pipeline_orchestrator.load_data(request.data_path)
        
        # Make predictions
        predictions = pipeline_orchestrator.predict(df)
        
        # Save predictions
        predictions.write.mode("overwrite").parquet(request.output_path)
        
        return {
            "status": "completed",
            "output_path": request.output_path,
            "record_count": predictions.count()
        }
    
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get training job status"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return training_jobs[job_id]


@app.get("/jobs")
async def list_jobs():
    """List all training jobs"""
    return {
        "jobs": training_jobs,
        "count": len(training_jobs)
    }


@app.get("/models")
async def list_models():
    """List available models from MLflow"""
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.list_experiments()
        
        models = []
        for exp in experiments:
            runs = client.search_runs(exp.experiment_id)
            for run in runs:
                models.append({
                    "run_id": run.info.run_id,
                    "experiment_id": exp.experiment_id,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "start_time": run.info.start_time
                })
        
        return {"models": models, "count": len(models)}
    
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """Get API metrics"""
    return {
        "total_training_jobs": len(training_jobs),
        "completed_jobs": sum(1 for j in training_jobs.values() if j["status"] == "completed"),
        "failed_jobs": sum(1 for j in training_jobs.values() if j["status"] == "failed"),
        "timestamp": datetime.now().isoformat()
    }


@app.delete("/models/{model_path}")
async def delete_model(model_path: str):
    """Delete a model"""
    try:
        # In production, implement actual model deletion
        return {
            "status": "deleted",
            "model_path": model_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
