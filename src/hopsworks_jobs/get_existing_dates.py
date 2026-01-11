"""
PySpark script to get existing dates from the traffic feature group.

This script runs on Hopsworks server-side and writes the unique dates
to an HDFS file that can be downloaded.

Usage (via Jobs API):
    Pass lookback_date as argument: "2026-01-06"
    
Output:
    Resources/existing_dates.txt on HDFS (single file, one date per line)
"""
import os
import sys
from datetime import datetime

import hsfs


def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: get_existing_dates.py <lookback_date>")
        print("  lookback_date: YYYY-MM-DD format")
        sys.exit(1)
    
    lookback_date_str = sys.argv[1]
    lookback_dt = datetime.strptime(lookback_date_str, "%Y-%m-%d")
    
    print(f"Querying dates >= {lookback_date_str}")
    
    # Connect to feature store
    connection = hsfs.connection()
    fs = connection.get_feature_store()
    
    # Get the traffic feature group
    traffic_fg = fs.get_feature_group("skane_traffic", version=2)
    
    # Read with Spark - this is much more reliable than Arrow Flight
    # The filter is pushed down to Spark
    df = traffic_fg.select(["date"]).filter(
        traffic_fg["event_time"] >= lookback_dt
    ).read()
    
    # Get unique dates
    unique_dates = df.select("date").distinct().collect()
    dates_list = sorted([row["date"] for row in unique_dates])
    
    print(f"Found {len(dates_list)} unique dates: {dates_list}")
    
    # Get the project name from environment
    project_name = os.environ.get("HOPSWORKS_PROJECT_NAME", "occupancy")
    
    # Write to HDFS as a single file using HDFS API
    # We use the subprocess to call hdfs dfs command which is available in Hopsworks
    output_path = f"/Projects/{project_name}/Resources/existing_dates.txt"
    hdfs_full_path = f"hdfs://{output_path}"
    
    # Create the content
    content = "\n".join(dates_list)
    
    # Use Spark's Hadoop configuration to write the file
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    
    # Get Hadoop configuration and FileSystem
    hadoop_conf = spark._jsc.hadoopConfiguration()
    
    # Get the FileSystem and Path classes
    from py4j.java_gateway import java_import
    java_import(spark._jvm, 'org.apache.hadoop.fs.Path')
    java_import(spark._jvm, 'org.apache.hadoop.fs.FileSystem')
    
    fs_hdfs = spark._jvm.FileSystem.get(hadoop_conf)
    path = spark._jvm.Path(hdfs_full_path)
    
    # Delete if exists
    if fs_hdfs.exists(path):
        fs_hdfs.delete(path, True)
    
    # Create and write to the file
    output_stream = fs_hdfs.create(path)
    output_stream.write(content.encode('utf-8'))
    output_stream.close()
    
    print(f"Wrote {len(dates_list)} dates to {hdfs_full_path}")
    connection.close()


if __name__ == "__main__":
    main()
