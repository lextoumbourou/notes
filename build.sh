#!/bin/bash

ENV=local uv run pelican ./notes/ --output=output/ --debug
