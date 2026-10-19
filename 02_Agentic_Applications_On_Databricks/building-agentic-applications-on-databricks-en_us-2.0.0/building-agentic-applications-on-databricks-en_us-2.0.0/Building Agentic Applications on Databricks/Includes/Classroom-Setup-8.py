# Databricks notebook source
import sys, os
includes_dir = "./Includes"
config_version = "8"
sys.path.insert(0, os.path.abspath('.'))
config_path = f"{includes_dir}/config/config-{config_version}.yaml"

from Includes._lib.library_installer import install_libraries
install_libraries(config_path)

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os, sys
includes_dir = "./Includes"
config_version = "8"
sys.path.insert(0, os.path.abspath('.'))

config_path = f"{includes_dir}/config/config-{config_version}.yaml"

# COMMAND ----------

from Includes import setup_demo_environment

env = setup_demo_environment(
    config_path=config_path,
)
catalog_name = env["catalog_name"]
schema_name  = env["schema_name"]
username     = env["username"]