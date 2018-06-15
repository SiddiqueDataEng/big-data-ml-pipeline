"""
Unit Tests for ML Pipeline
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ml_pipeline import MLPipelineOrchestrator
from src.feature_engineering import FeatureEngineer


@pytest.fixture(scope="session")
def spark():
    """Create Spark session for testing"""
    spark = SparkSession.builder \
        .appName("ML Pipeline Tests") \
        .master("local[2]") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()
    
    yield spark
    spark.stop()


@pytest.fixture
def sample_data(spark):
    """Create sample training data"""
    schema = StructType([
        StructField("feature_1", DoubleType(), True),
        StructField("feature_2", DoubleType(), True),
        StructField("feature_3", DoubleType(), True),
        StructField("label", IntegerType(), True)
    ])
    
    data = [
        (1.0, 2.0, 3.0, 0),
        (2.0, 3.0, 4.0, 1),
        (3.0, 4.0, 5.0, 0),
        (4.0, 5.0, 6.0, 1),
        (5.0, 6.0, 7.0, 0),
        (6.0, 7.0, 8.0, 1),
        (7.0, 8.0, 9.0, 0),
        (8.0, 9.0, 10.0, 1),
    ]
    
    return spark.createDataFrame(data, schema)


@pytest.fixture
def config_file(tmp_path):
    """Create temporary config file"""
    config_content = """
spark:
  app_name: "Test Pipeline"
  master: "local[2]"

data:
  input_path: "test_data.parquet"
  output_path: "test_models/"

features:
  columns:
    - feature_1
    - feature_2
    - feature_3
  target: "label"

model:
  algorithm: "random_forest"
  task_type: "classification"
  hyperparameters:
    numTrees: 10
    maxDepth: 5
"""
    
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(config_content)
    return str(config_path)


class TestMLPipeline:
    """Test ML Pipeline functionality"""
    
    def test_pipeline_initialization(self, config_file):
        """Test pipeline initialization"""
        pipeline = MLPipelineOrchestrator(config_file)
        assert pipeline is not None
        assert pipeline.spark is not None
        assert pipeline.config is not None
        pipeline.stop()
    
    def test_feature_pipeline_creation(self, config_file, sample_data):
        """Test feature pipeline creation"""
        pipeline = MLPipelineOrchestrator(config_file)
        feature_pipeline = pipeline.build_feature_pipeline(sample_data)
        
        assert feature_pipeline is not None
        assert len(feature_pipeline.getStages()) > 0
        
        pipeline.stop()
    
    def test_model_training_random_forest(self, config_file, sample_data):
        """Test Random Forest model training"""
        pipeline = MLPipelineOrchestrator(config_file)
        
        model = pipeline.train(sample_data, "random_forest", "classification")
        
        assert model is not None
        assert pipeline.model is not None
        
        pipeline.stop()
    
    def test_model_training_logistic_regression(self, config_file, sample_data):
        """Test Logistic Regression model training"""
        pipeline = MLPipelineOrchestrator(config_file)
        
        model = pipeline.train(sample_data, "logistic_regression", "classification")
        
        assert model is not None
        
        pipeline.stop()
    
    def test_model_prediction(self, config_file, sample_data):
        """Test model prediction"""
        pipeline = MLPipelineOrchestrator(config_file)
        
        # Train model
        pipeline.train(sample_data, "random_forest", "classification")
        
        # Make predictions
        predictions = pipeline.predict(sample_data)
        
        assert predictions is not None
        assert "prediction" in predictions.columns
        assert predictions.count() == sample_data.count()
        
        pipeline.stop()
    
    def test_model_evaluation(self, config_file, sample_data):
        """Test model evaluation"""
        pipeline = MLPipelineOrchestrator(config_file)
        
        # Train model
        pipeline.train(sample_data, "random_forest", "classification")
        
        # Make predictions
        predictions = pipeline.predict(sample_data)
        
        # Evaluate
        metrics = pipeline.evaluate(predictions, "classification")
        
        assert metrics is not None
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        
        pipeline.stop()
    
    def test_model_save_load(self, config_file, sample_data, tmp_path):
        """Test model save and load"""
        pipeline = MLPipelineOrchestrator(config_file)
        
        # Train model
        pipeline.train(sample_data, "random_forest", "classification")
        
        # Save model
        model_path = str(tmp_path / "test_model")
        pipeline.save_model(model_path)
        
        # Load model
        loaded_model = pipeline.load_model(model_path)
        
        assert loaded_model is not None
        
        # Make predictions with loaded model
        predictions = pipeline.predict(sample_data)
        assert predictions is not None
        
        pipeline.stop()


class TestFeatureEngineering:
    """Test Feature Engineering functionality"""
    
    def test_time_features(self, spark):
        """Test time feature creation"""
        from pyspark.sql.functions import current_timestamp
        
        df = spark.range(10).withColumn("timestamp", current_timestamp())
        
        engineer = FeatureEngineer(spark)
        result = engineer.create_time_features(df, "timestamp")
        
        assert "hour" in result.columns
        assert "day_of_week" in result.columns
        assert "is_weekend" in result.columns
    
    def test_ratio_features(self, spark):
        """Test ratio feature creation"""
        data = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        df = spark.createDataFrame(data, ["col1", "col2"])
        
        engineer = FeatureEngineer(spark)
        result = engineer.create_ratio_features(df, ["col1"], ["col2"])
        
        assert "col1_to_col2_ratio" in result.columns
    
    def test_interaction_features(self, spark):
        """Test interaction feature creation"""
        data = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
        df = spark.createDataFrame(data, ["col1", "col2", "col3"])
        
        engineer = FeatureEngineer(spark)
        result = engineer.create_interaction_features(df, ["col1", "col2"])
        
        assert "col1_x_col2" in result.columns
    
    def test_missing_value_handling(self, spark):
        """Test missing value handling"""
        data = [(1.0, 2.0), (None, 3.0), (4.0, None)]
        df = spark.createDataFrame(data, ["col1", "col2"])
        
        engineer = FeatureEngineer(spark)
        result = engineer.handle_missing_values(df, strategy="mean")
        
        assert result.filter(result.col1.isNull()).count() == 0
    
    def test_outlier_detection(self, spark):
        """Test outlier detection"""
        data = [(1.0,), (2.0,), (3.0,), (100.0,)]  # 100.0 is outlier
        df = spark.createDataFrame(data, ["value"])
        
        engineer = FeatureEngineer(spark)
        result = engineer.detect_outliers(df, ["value"], threshold=2.0)
        
        assert "value_is_outlier" in result.columns
        outliers = result.filter(result.value_is_outlier == 1).count()
        assert outliers > 0
    
    def test_frequency_encoding(self, spark):
        """Test frequency encoding"""
        data = [("A",), ("B",), ("A",), ("C",), ("A",)]
        df = spark.createDataFrame(data, ["category"])
        
        engineer = FeatureEngineer(spark)
        result = engineer.create_frequency_encoding(df, ["category"])
        
        assert "category_freq" in result.columns
        freq_a = result.filter(result.category == "A").select("category_freq").first()[0]
        assert freq_a == 3


class TestModelAlgorithms:
    """Test different ML algorithms"""
    
    def test_random_forest_classifier(self, config_file, sample_data):
        """Test Random Forest Classifier"""
        pipeline = MLPipelineOrchestrator(config_file)
        model = pipeline.get_model("random_forest", "classification")
        
        assert model is not None
        assert hasattr(model, 'numTrees')
        
        pipeline.stop()
    
    def test_logistic_regression(self, config_file, sample_data):
        """Test Logistic Regression"""
        pipeline = MLPipelineOrchestrator(config_file)
        model = pipeline.get_model("logistic_regression", "classification")
        
        assert model is not None
        assert hasattr(model, 'maxIter')
        
        pipeline.stop()
    
    def test_gbt_classifier(self, config_file, sample_data):
        """Test GBT Classifier"""
        pipeline = MLPipelineOrchestrator(config_file)
        model = pipeline.get_model("gbt", "classification")
        
        assert model is not None
        assert hasattr(model, 'maxIter')
        
        pipeline.stop()
    
    def test_linear_regression(self, config_file, sample_data):
        """Test Linear Regression"""
        pipeline = MLPipelineOrchestrator(config_file)
        model = pipeline.get_model("linear_regression", "regression")
        
        assert model is not None
        assert hasattr(model, 'maxIter')
        
        pipeline.stop()
    
    def test_kmeans_clustering(self, config_file, sample_data):
        """Test K-Means Clustering"""
        pipeline = MLPipelineOrchestrator(config_file)
        model = pipeline.get_model("kmeans", "clustering")
        
        assert model is not None
        assert hasattr(model, 'k')
        
        pipeline.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
