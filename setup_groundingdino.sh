#!/bin/bash
# Setup script for GroundingDINO object detection

echo "Setting up GroundingDINO..."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Warning: .venv not found. Please create virtual environment first:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    exit 1
fi

# 1. Install GroundingDINO
echo "Installing groundingdino-py..."
pip install groundingdino-py

# 2. Create cache directory
echo "Creating cache directory..."
mkdir -p cache

# 3. Download model files (SwinB version - better accuracy)
echo "Downloading model weights (this may take a few minutes)..."
cd cache

# Download SwinB config
echo "  Downloading config file..."
curl -L -o GroundingDINO_SwinB_cfg.py https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinB_cfg.py

# Download SwinB checkpoint
echo "  Downloading model checkpoint (~600MB)..."
curl -L -o groundingdino_swinb_cogcoor.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

cd ..

echo ""
echo "GroundingDINO setup complete!"
echo ""
echo "Model files installed in ./cache/:"
ls -lh cache/

echo ""
echo "Your agent can now use precise object detection!"
echo "Try running your Kingdom Rush task again - the agent will now be able to:"
echo "  - Detect the red flag icon precisely"
echo "  - Find tower building sites"
echo "  - Identify enemy units"
echo "  - Click any visual element accurately"
