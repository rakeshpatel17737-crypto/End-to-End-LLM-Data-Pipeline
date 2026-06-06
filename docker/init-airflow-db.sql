-- Create the separate database Airflow uses for its metadata.
-- Runs once on first Postgres init, before 01-init.sql.
CREATE DATABASE airflow;
