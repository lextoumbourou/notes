#!/bin/bash

ENV=local uv run python -m pelican ./notes/ --output=output/
