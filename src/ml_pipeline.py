"""
Big Data ML Pipeline - Main Orchestration
Distributed machine learning pipeline using Apache Spark MLlib
"""

import logging
from typing import Dict, Any, Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier
from pyspark.ml.regression import RandomForestRegressor, LinearRegression, GBTRegressor
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import mlflow
import mlflow.spark
from datetime import datetime
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLPipelineOrchestrator:
    """Main orchestrator for distributed ML pipeline"""
    
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.spark = self._create_spark_session()
        self.model = None
        self.pipeline = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load pipeline configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_spark_session(self) -> SparkSession:
        """Create Spark session with optimized settings"""
        spark_config = self.config.get('spark', {})
        
        builder = SparkSession.builder \
            .appName(spark_config.get('app_name', 'ML Pipeline')) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        
        if 'master' in spark_config:
            builder = builder.master(spark_config['master'])
        
        return builder.getOrCreate()
    
    def load_data(self, data_path: Optional[str] = None) -> DataFrame:
        """Load training data from various sources"""
        path = data_path or self.config['data']['input_path']
        
        logger.info(f"Loading data from {path}")
        
        if path.endswith('.parquet'):
            df = self.spark.read.parquet(path)
        elif path.endswith('.csv'):
            df = self.spark.read.csv(path, header=True, inferSchema=True)
        elif path.endswith('.delta'):
            df = self.spark.read.format("delta").load(path)
        else:
            # Default to Delta Lake
            df = self.spark.read.format("delta").load(path)
        
        logger.info(f"Loaded {df.count()} rows with {len(df.columns)} columns")
        return df
    
    def build_feature_pipeline(self, df: DataFrame) -> Pipeline:
        """Build feature engineering pipeline"""
        feature_config = self.config.get('features', {})
        feature_cols = feature_config.get('columns', df.columns[:-1])
        target_col = feature_config.get('target', 'label')
        
        stages = []
        
        # Handle categorical features
        categorical_cols = [f.name for f in df.schema.fields 
                          if f.dataType.simpleString() == 'string' and f.name != target_col]
        
        for col in categorical_cols:
            indexer = StringIndexer(inputCol=col, outputCol=f"{col}_indexed", handleInvalid="keep")
            stages.append(indexer)
        
        # Assemble features
        numeric_cols = [f.name for f in df.schema.fields 
                       if f.dataType.simpleString() in ['int', 'double', 'float'] and f.name != target_col]
        indexed_cols = [f"{col}_indexed" for col in categorical_cols]
        
        assembler = VectorAssembler(
            inputCols=numeric_cols + indexed_cols,
            outputCol="features_raw",
            handleInvalid="skip"
        )
        stages.append(assembler)
        
        # Scale features
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=False
        )
        stages.append(scaler)
        
        return Pipeline(stages=stages)
    
    def get_model(self, algorithm: str, task_type: str = 'classification'):
        """Get ML model based on algorithm and task type"""
        model_config = self.config.get('model', {})
        params = model_config.get('hyperparameters', {})
        
        if task_type == 'classification':
            if algorithm == 'random_forest':
                return RandomForestClassifier(
                    featuresCol="features",
                    labelCol="label",
                    numTrees=params.get('numTrees', 100),
                    maxDepth=params.get('maxDepth', 10),
                    seed=42
                )
            elif algorithm == 'logistic_regression':
                return LogisticRegression(
                    featuresCol="features",
                    labelCol="label",
                    maxIter=params.get('maxIter', 100),
                    regParam=params.get('regParam', 0.01)
                )
            elif algorithm == 'gbt':
                return GBTClassifier(
                    featuresCol="features",
                    labelCol="label",
                    maxIter=params.get('maxIter', 100),
                    maxDepth=params.get('maxDepth', 5)
                )
        
        elif task_type == 'regression':
            if algorithm == 'random_forest':
                return RandomForestRegressor(
                    featuresCol="features",
                    labelCol="label",
                    numTrees=params.get('numTrees', 100),
                    maxDepth=params.get('maxDepth', 10)
                )
            elif algorithm == 'linear_regression':
                return LinearRegression(
                    featuresCol="features",
                    labelCol="label",
                    maxIter=params.get('maxIter', 100)
                )
            elif algorithm == 'gbt':
                return GBTRegressor(
                    featuresCol="features",
                    labelCol="label",
                    maxIter=params.get('maxIter', 100)
                )
        
        elif task_type == 'clustering':
            if algorithm == 'kmeans':
                return KMeans(
                    featuresCol="features",
                    k=params.get('k', 5),
                    seed=42
                )
            elif algorithm == 'bisecting_kmeans':
                return BisectingKMeans(
                    featuresCol="features",
                    k=params.get('k', 5),
                    seed=42
                )
        
        raise ValueError(f"Unknown algorithm: {algorithm} for task: {task_type}")
    
    def train(self, df: DataFrame, algorithm: str, task_type: str = 'classification') -> PipelineModel:
        """Train ML model with feature pipeline"""
        logger.info(f"Training {algorithm} model for {task_type}")
        
        # Split data
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
        
        # Build feature pipeline
        feature_pipeline = self.build_feature_pipeline(train_df)
        
        # Get model
        model = self.get_model(algorithm, task_type)
        
        # Create full pipeline
        self.pipeline = Pipeline(stages=feature_pipeline.getStages() + [model])
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params(self.config.get('model', {}).get('hyperparameters', {}))
            mlflow.log_param("algorithm", algorithm)
            mlflow.log_param("task_type", task_type)
            
            # Train model
            logger.info("Training model...")
            self.model = self.pipeline.fit(train_df)
            
            # Evaluate
            predictions = self.model.transform(test_df)
            metrics = self.evaluate(predictions, task_type)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.spark.log_model(self.model, "model")
            
            logger.info(f"Training complete. Metrics: {metrics}")
        
        return self.model
    
    def evaluate(self, predictions: DataFrame, task_type: str = 'classification') -> Dict[str, float]:
        """Evaluate model performance"""
        metrics = {}
        
        if task_type == 'classification':
            # Binary classification metrics
            evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
            metrics['auc_roc'] = evaluator.evaluate(predictions)
            
            evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderPR")
            metrics['auc_pr'] = evaluator.evaluate(predictions)
            
            # Multiclass metrics
            mc_evaluator = MulticlassClassificationEvaluator(labelCol="label")
            metrics['accuracy'] = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "accuracy"})
            metrics['f1'] = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "f1"})
            metrics['precision'] = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "weightedPrecision"})
            metrics['recall'] = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "weightedRecall"})
        
        return metrics
    
    def hyperparameter_tuning(self, df: DataFrame, algorithm: str, task_type: str = 'classification'):
        """Perform hyperparameter tuning with cross-validation"""
        logger.info("Starting hyperparameter tuning...")
        
        # Build feature pipeline
        feature_pipeline = self.build_feature_pipeline(df)
        
        # Get model
        model = self.get_model(algorithm, task_type)
        
        # Create pipeline
        pipeline = Pipeline(stages=feature_pipeline.getStages() + [model])
        
        # Parameter grid
        paramGrid = ParamGridBuilder()
        
        if algorithm == 'random_forest':
            paramGrid = paramGrid \
                .addGrid(model.numTrees, [50, 100, 200]) \
                .addGrid(model.maxDepth, [5, 10, 15]) \
                .build()
        elif algorithm == 'logistic_regression':
            paramGrid = paramGrid \
                .addGrid(model.regParam, [0.01, 0.1, 1.0]) \
                .addGrid(model.elasticNetParam, [0.0, 0.5, 1.0]) \
                .build()
        else:
            paramGrid = paramGrid.build()
        
        # Cross-validator
        evaluator = BinaryClassificationEvaluator(labelCol="label")
        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=paramGrid,
            evaluator=evaluator,
            numFolds=3,
            parallelism=4
        )
        
        # Fit
        self.model = cv.fit(df)
        
        logger.info("Hyperparameter tuning complete")
        return self.model
    
    def save_model(self, path: Optional[str] = None):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        save_path = path or self.config['data']['output_path']
        logger.info(f"Saving model to {save_path}")
        self.model.write().overwrite().save(save_path)
    
    def load_model(self, path: str) -> PipelineModel:
        """Load saved model"""
        logger.info(f"Loading model from {path}")
        self.model = PipelineModel.load(path)
        return self.model
    
    def predict(self, df: DataFrame) -> DataFrame:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        return self.model.transform(df)
    
    def stop(self):
        """Stop Spark session"""
        if self.spark:
            self.spark.stop()


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Big Data ML Pipeline')
    parser.add_argument('--config', type=str, default='config/pipeline_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, help='Path to training data')
    parser.add_argument('--algorithm', type=str, default='random_forest',
                       choices=['random_forest', 'logistic_regression', 'gbt', 'linear_regression', 'kmeans'])
    parser.add_argument('--task', type=str, default='classification',
                       choices=['classification', 'regression', 'clustering'])
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MLPipelineOrchestrator(args.config)
    
    try:
        # Load data
        df = pipeline.load_data(args.data)
        
        # Train model
        if args.tune:
            model = pipeline.hyperparameter_tuning(df, args.algorithm, args.task)
        else:
            model = pipeline.train(df, args.algorithm, args.task)
        
        # Save model
        pipeline.save_model()
        
        logger.info("Pipeline execution complete!")
        
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
