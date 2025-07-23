#!/bin/bash
# Setup script for ECG simulator testing environment

echo "🧪 Setting up ECG Simulator Test Environment"
echo "============================================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Consider activating your virtual environment first:"
    echo "   source venv/bin/activate"
    echo ""
fi

# Find the correct pip command
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    echo "❌ Neither pip nor pip3 found. Please install pip."
    exit 1
fi

echo "Using pip command: $PIP_CMD"

# Install test dependencies
echo "📦 Installing test dependencies..."
$PIP_CMD install -r requirements-test.txt

if [ $? -eq 0 ]; then
    echo "✅ Test dependencies installed successfully"
else
    echo "❌ Failed to install test dependencies"
    exit 1
fi

# Install main dependencies if needed
echo ""
echo "📦 Installing main dependencies..."
$PIP_CMD install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Main dependencies installed successfully"
else
    echo "❌ Failed to install main dependencies"
    exit 1
fi

echo ""
echo "🚀 Setup complete! You can now run tests:"
echo ""
echo "  # Run all tests"
echo "  python3 run_tests.py"
echo ""
echo "  # Run quick tests only"
echo "  python3 run_tests.py --quick"
echo ""
echo "  # Run medical accuracy tests"
echo "  python3 run_tests.py --medical"
echo ""
echo "  # Run with coverage report"
echo "  python3 run_tests.py --coverage --html"
echo ""
echo "📊 View the test documentation: tests/README.md"