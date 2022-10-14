# TESTING SCRIPT
# Run this script to test application
# 
# Dependencies:
# - pytest
# - pytest-cov
# - coverage-badge
# - xfvb

# run the tests (headless, to allow running on servers)
xvfb-run pytest --cov src/minerl3161  --cov-report term-missing tests/

# generate coverage SVG
coverage-badge -o coverage.svg -f