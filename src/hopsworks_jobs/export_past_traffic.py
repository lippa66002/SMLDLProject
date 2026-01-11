"""
PySpark script to export past traffic data for batch inference.

This script runs on Hopsworks server-side, queries the traffic feature group
for a date range, and saves to a parquet file that can be downloaded.

Usage (via Jobs API):
    Pass arguments: "<min_date> <max_date>"
    Example: "2026-01-08 2026-01-10"
    
Output:
    Resources/past_traffic_data.parquet on HDFS
"""
import os
import sys
from datetime import datetime

import hsfs


def main():
    # Parse arguments
    if len(sys.argv) < 3:
        print("Usage: export_past_traffic.py <min_date> <max_date>")
        print("  min_date: YYYY-MM-DD format")
        print("  max_date: YYYY-MM-DD format")
        sys.exit(1)
    
    min_date_str = sys.argv[1]
    max_date_str = sys.argv[2]
    
    min_dt = datetime.strptime(min_date_str, "%Y-%m-%d")
    max_dt = datetime.strptime(max_date_str, "%Y-%m-%d")
    
    print(f"Querying traffic data from {min_date_str} to {max_date_str}")
    
    # Connect to feature store
    connection = hsfs.connection()
    fs = connection.get_feature_store()
    
    # Get the feature groups
    traffic_fg = fs.get_feature_group("skane_traffic", version=2)
    weather_fg = fs.get_feature_group("skane_weather", version=2)
    calendar_fg = fs.get_feature_group("sweden_calendar", version=2)
    
    # Read traffic data with filter (only min_date, no max since it's all historical)
    print("Reading traffic data...")
    traffic_df = traffic_fg.filter(
        traffic_fg["event_time"] >= min_dt
    ).read()
    
    if traffic_df.count() == 0:
        print("No traffic data found for the specified date range.")
        connection.close()
        sys.exit(0)
    
    print(f"Found {traffic_df.count()} traffic rows")
    
    # Read weather data (only min_date filter)
    print("Reading weather data...")
    weather_df = weather_fg.filter(
        weather_fg["event_time"] >= min_dt
    ).read()
    print(f"Found {weather_df.count()} weather rows")
    
    # Read calendar data (only min_date filter)
    print("Reading calendar data...")
    calendar_df = calendar_fg.filter(
        calendar_fg["event_time"] >= min_dt
    ).read()
    print(f"Found {calendar_df.count()} calendar rows")
    
    # Join the dataframes
    # First merge traffic with calendar on date
    print("Joining dataframes...")
    
    # Rename event_time columns to avoid conflicts
    traffic_df = traffic_df.withColumnRenamed("event_time", "traffic_event_time")
    weather_df = weather_df.withColumnRenamed("event_time", "weather_event_time")
    calendar_df = calendar_df.withColumnRenamed("event_time", "calendar_event_time")
    
    # Join traffic with calendar on date
    merged_df = traffic_df.join(calendar_df, on="date", how="left")
    
    # Join with weather on date and hour
    merged_df = merged_df.join(weather_df, on=["date", "hour"], how="left")
    
    # Select and rename columns for output
    # Keep all columns we need for inference
    result_df = merged_df.select(
        "date", "hour", "route_id", "direction_id",
        "avg_occupancy", "max_occupancy",
        # Calendar features
        "month", "weekday", "is_weekend", "is_holiday_se", "is_workday_se",
        # Weather features
        "temperature_2m", "precipitation", "windspeed_10m", "cloudcover",
        "prev_temperature_2m", "prev_precipitation", "prev_windspeed_10m", "prev_cloudcover"
    )
    
    print(f"Merged dataframe has {result_df.count()} rows")
    
    # Get the project name from environment
    project_name = os.environ.get("HOPSWORKS_PROJECT_NAME", "occupancy")
    
    # Write to HDFS as parquet (single file) using Hadoop FileSystem API directly
    # This approach creates a true single file, not a directory
    output_path = f"/Projects/{project_name}/Resources/past_traffic_data.parquet"
    hdfs_full_path = f"hdfs://{output_path}"
    
    print(f"Writing to {hdfs_full_path}...")
    
    # Convert to pandas for single-file write
    pandas_df = result_df.toPandas()
    
    # Write parquet to a local temp file first
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        local_parquet_path = tmp.name
    
    pandas_df.to_parquet(local_parquet_path, index=False)
    
    # Read the local file bytes
    with open(local_parquet_path, 'rb') as f:
        parquet_bytes = f.read()
    
    # Clean up local file
    os.unlink(local_parquet_path)
    
    # Use Spark's Hadoop FileSystem API to write directly to HDFS
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    
    # Get Hadoop configuration and FileSystem
    hadoop_conf = spark._jsc.hadoopConfiguration()
    
    # Import FileSystem and Path classes
    from py4j.java_gateway import java_import
    java_import(spark._jvm, 'org.apache.hadoop.fs.Path')
    java_import(spark._jvm, 'org.apache.hadoop.fs.FileSystem')
    
    fs_hdfs = spark._jvm.FileSystem.get(hadoop_conf)
    path = spark._jvm.Path(hdfs_full_path)
    
    # Delete if exists (could be file or directory)
    if fs_hdfs.exists(path):
        print(f"Removing existing path at {hdfs_full_path}")
        fs_hdfs.delete(path, True)  # True = recursive delete if directory
    
    # Create and write to the file
    output_stream = fs_hdfs.create(path)
    output_stream.write(parquet_bytes)
    output_stream.close()
    
    print(f"Successfully wrote {len(pandas_df)} rows to {hdfs_full_path}")
    connection.close()


if __name__ == "__main__":
    main()
