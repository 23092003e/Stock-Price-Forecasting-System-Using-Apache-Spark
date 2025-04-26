# utils.py

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, mean, stddev, to_date, date_format
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler


def initialize_spark():
    spark = SparkSession.builder.appName("Netflix Stock Price Prediction").getOrCreate()
    return spark

def load_data(spark, file_path):
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    return df
def clean_data(df: DataFrame) -> DataFrame:
    df = df.dropna()
    return df

def convert_data_format(df: DataFrame) -> DataFrame:
    df = df.withColumn("Date", col("Date").cast("timestamp"))
    df = df.withColumn("Date", to_date(col("Date"), "yyyy-MM-dd"))
    return df

def feature_engineering(df: DataFrame) -> DataFrame:
    window = Window.partitionBy("Date").orderBy("Date")
    



