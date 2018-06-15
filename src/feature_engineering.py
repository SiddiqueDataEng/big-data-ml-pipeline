"""
Feature Engineering Module
Advanced feature transformations for ML pipelines
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, lit, udf, array, explode
from pyspark.sql.types import DoubleType, ArrayType
from pyspark.ml.feature import (
    Bucketizer, QuantileDiscretizer, OneHotEncoder,
    PCA, PolynomialExpansion, Interaction, SQLTransformer
)
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Advanced feature engineering operations"""
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def create_time_features(self, df: DataFrame, timestamp_col: str) -> DataFrame:
        """Extract time-based features from timestamp"""
        logger.info(f"Creating time features from {timestamp_col}")
        
        df = df.withColumn("hour", col(timestamp_col).cast("timestamp").cast("long") % 86400 / 3600)
        df = df.withColumn("day_of_week", (col(timestamp_col).cast("timestamp").cast("long") / 86400) % 7)
        df = df.withColumn("day_of_month", (col(timestamp_col).cast("timestamp").cast("long") / 86400) % 31 + 1)
        df = df.withColumn("is_weekend", when(col("day_of_week").isin([5, 6]), 1).otherwise(0))
        
        return df
    
    def create_aggregation_features(self, df: DataFrame, group_cols: List[str], 
                                   agg_cols: List[str]) -> DataFrame:
        """Create aggregation features"""
        logger.info(f"Creating aggregation features for {group_cols}")
        
        from pyspark.sql.functions import mean, stddev, min as spark_min, max as spark_max, count
        
        agg_exprs = []
        for agg_col in agg_cols:
            agg_exprs.extend([
                mean(agg_col).alias(f"{agg_col}_mean"),
                stddev(agg_col).alias(f"{agg_col}_std"),
                spark_min(agg_col).alias(f"{agg_col}_min"),
                spark_max(agg_col).alias(f"{agg_col}_max"),
                count(agg_col).alias(f"{agg_col}_count")
            ])
        
        agg_df = df.groupBy(group_cols).agg(*agg_exprs)
        
        # Join back to original dataframe
        result_df = df.join(agg_df, on=group_cols, how='left')
        
        return result_df
    
    def create_ratio_features(self, df: DataFrame, numerator_cols: List[str], 
                            denominator_cols: List[str]) -> DataFrame:
        """Create ratio features"""
        logger.info("Creating ratio features")
        
        for num_col in numerator_cols:
            for denom_col in denominator_cols:
                if num_col != denom_col:
                    ratio_col = f"{num_col}_to_{denom_col}_ratio"
                    df = df.withColumn(
                        ratio_col,
                        when(col(denom_col) != 0, col(num_col) / col(denom_col)).otherwise(0)
                    )
        
        return df
    
    def create_interaction_features(self, df: DataFrame, feature_cols: List[str]) -> DataFrame:
        """Create interaction features between columns"""
        logger.info("Creating interaction features")
        
        for i, col1 in enumerate(feature_cols):
            for col2 in feature_cols[i+1:]:
                interaction_col = f"{col1}_x_{col2}"
                df = df.withColumn(interaction_col, col(col1) * col(col2))
        
        return df
    
    def create_binning_features(self, df: DataFrame, numeric_cols: List[str], 
                               num_bins: int = 10) -> DataFrame:
        """Create binned features from numeric columns"""
        logger.info(f"Creating binning features with {num_bins} bins")
        
        for numeric_col in numeric_cols:
            discretizer = QuantileDiscretizer(
                numBuckets=num_bins,
                inputCol=numeric_col,
                outputCol=f"{numeric_col}_binned",
                handleInvalid="keep"
            )
            df = discretizer.fit(df).transform(df)
        
        return df
    
    def create_polynomial_features(self, df: DataFrame, feature_col: str, 
                                  degree: int = 2) -> DataFrame:
        """Create polynomial features"""
        logger.info(f"Creating polynomial features of degree {degree}")
        
        poly_expansion = PolynomialExpansion(
            degree=degree,
            inputCol=feature_col,
            outputCol=f"{feature_col}_poly"
        )
        
        return poly_expansion.transform(df)
    
    def create_pca_features(self, df: DataFrame, feature_col: str, 
                          n_components: int = 10) -> DataFrame:
        """Create PCA features for dimensionality reduction"""
        logger.info(f"Creating PCA features with {n_components} components")
        
        pca = PCA(
            k=n_components,
            inputCol=feature_col,
            outputCol=f"{feature_col}_pca"
        )
        
        model = pca.fit(df)
        return model.transform(df)
    
    def handle_missing_values(self, df: DataFrame, strategy: str = 'mean') -> DataFrame:
        """Handle missing values in dataframe"""
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        from pyspark.ml.feature import Imputer
        
        numeric_cols = [f.name for f in df.schema.fields 
                       if f.dataType.simpleString() in ['int', 'double', 'float']]
        
        if strategy in ['mean', 'median']:
            imputer = Imputer(
                strategy=strategy,
                inputCols=numeric_cols,
                outputCols=[f"{col}_imputed" for col in numeric_cols]
            )
            df = imputer.fit(df).transform(df)
            
            # Replace original columns
            for col_name in numeric_cols:
                df = df.drop(col_name).withColumnRenamed(f"{col_name}_imputed", col_name)
        
        elif strategy == 'drop':
            df = df.dropna()
        
        return df
    
    def detect_outliers(self, df: DataFrame, numeric_cols: List[str], 
                       threshold: float = 3.0) -> DataFrame:
        """Detect outliers using z-score method"""
        logger.info(f"Detecting outliers with threshold {threshold}")
        
        from pyspark.sql.functions import mean, stddev
        
        for col_name in numeric_cols:
            stats = df.select(mean(col(col_name)).alias('mean'), 
                            stddev(col(col_name)).alias('std')).first()
            
            if stats['std'] and stats['std'] > 0:
                df = df.withColumn(
                    f"{col_name}_is_outlier",
                    when(
                        (col(col_name) - stats['mean']).abs() > threshold * stats['std'],
                        1
                    ).otherwise(0)
                )
        
        return df
    
    def create_lag_features(self, df: DataFrame, partition_cols: List[str],
                          order_col: str, value_cols: List[str], 
                          lags: List[int] = [1, 7, 30]) -> DataFrame:
        """Create lag features for time series"""
        logger.info(f"Creating lag features: {lags}")
        
        from pyspark.sql.window import Window
        from pyspark.sql.functions import lag
        
        window_spec = Window.partitionBy(partition_cols).orderBy(order_col)
        
        for value_col in value_cols:
            for lag_period in lags:
                df = df.withColumn(
                    f"{value_col}_lag_{lag_period}",
                    lag(col(value_col), lag_period).over(window_spec)
                )
        
        return df
    
    def create_rolling_features(self, df: DataFrame, partition_cols: List[str],
                              order_col: str, value_cols: List[str],
                              windows: List[int] = [7, 30]) -> DataFrame:
        """Create rolling window features"""
        logger.info(f"Creating rolling features: {windows}")
        
        from pyspark.sql.window import Window
        from pyspark.sql.functions import avg, sum as spark_sum
        
        for value_col in value_cols:
            for window_size in windows:
                window_spec = Window.partitionBy(partition_cols) \
                    .orderBy(order_col) \
                    .rowsBetween(-window_size, 0)
                
                df = df.withColumn(
                    f"{value_col}_rolling_mean_{window_size}",
                    avg(col(value_col)).over(window_spec)
                )
                df = df.withColumn(
                    f"{value_col}_rolling_sum_{window_size}",
                    spark_sum(col(value_col)).over(window_spec)
                )
        
        return df
    
    def create_frequency_encoding(self, df: DataFrame, categorical_cols: List[str]) -> DataFrame:
        """Create frequency encoding for categorical variables"""
        logger.info("Creating frequency encoding")
        
        from pyspark.sql.functions import count
        
        for cat_col in categorical_cols:
            freq_df = df.groupBy(cat_col).agg(count("*").alias(f"{cat_col}_freq"))
            df = df.join(freq_df, on=cat_col, how='left')
        
        return df
    
    def create_target_encoding(self, df: DataFrame, categorical_cols: List[str],
                             target_col: str) -> DataFrame:
        """Create target encoding for categorical variables"""
        logger.info("Creating target encoding")
        
        from pyspark.sql.functions import mean
        
        for cat_col in categorical_cols:
            target_df = df.groupBy(cat_col).agg(
                mean(col(target_col)).alias(f"{cat_col}_target_mean")
            )
            df = df.join(target_df, on=cat_col, how='left')
        
        return df


def create_feature_pipeline(df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Create complete feature engineering pipeline"""
    engineer = FeatureEngineer(df.sparkSession)
    
    # Apply transformations based on config
    if 'time_features' in config:
        df = engineer.create_time_features(df, config['time_features']['timestamp_col'])
    
    if 'aggregations' in config:
        df = engineer.create_aggregation_features(
            df,
            config['aggregations']['group_cols'],
            config['aggregations']['agg_cols']
        )
    
    if 'ratios' in config:
        df = engineer.create_ratio_features(
            df,
            config['ratios']['numerator_cols'],
            config['ratios']['denominator_cols']
        )
    
    if 'missing_values' in config:
        df = engineer.handle_missing_values(df, config['missing_values']['strategy'])
    
    return df
