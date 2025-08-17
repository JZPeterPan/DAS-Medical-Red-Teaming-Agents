#!/bin/bash

# Medical Red-Teaming Framework - Environment Setup Script
# This script sets up the complete environment for the medical red-teaming framework

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        REQUIRED_VERSION="3.10"
        
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
            print_success "Python $PYTHON_VERSION found (>= 3.10 required)"
            return 0
        else
            print_error "Python $PYTHON_VERSION found, but 3.10+ is required"
            return 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.10+ first."
        return 1
    fi
}

# Function to check pip
check_pip() {
    if command_exists pip3; then
        print_success "pip3 found"
        return 0
    elif command_exists pip; then
        print_success "pip found"
        return 0
    else
        print_error "pip not found. Please install pip first."
        return 1
    fi
}

# Function to install Python packages
install_python_packages() {
    print_status "Installing Python packages..."
    
    # Check if requirements.txt exists
    if [ -f "requirements.txt" ]; then
        print_status "Installing packages from requirements.txt..."
        if command_exists pip3; then
            pip3 install -r requirements.txt --quiet
        else
            pip install -r requirements.txt --quiet
        fi
        print_success "All packages from requirements.txt installed successfully"
    else
        print_warning "requirements.txt not found, installing core packages individually..."
        
        # Install packages that need pip
        PIP_PACKAGES=(
            "pandas"
            "pydantic"
            "openai"
            "anthropic"
            "google-genai"
            "tqdm"
        )
        
        for package in "${PIP_PACKAGES[@]}"; do
            print_status "Installing $package..."
            if command_exists pip3; then
                pip3 install "$package" --quiet
            else
                pip install "$package" --quiet
            fi
            print_success "$package installed"
        done
    fi
    
    print_success "All Python packages installed successfully"
}

# Function to create virtual environment (optional)
create_virtual_environment() {
    print_status "Checking for virtual environment..."
    
    if [ -d "venv" ] || [ -d ".venv" ]; then
        print_warning "Virtual environment already exists"
        return 0
    fi
    
    read -p "Do you want to create a virtual environment? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
        print_status "To activate the virtual environment, run:"
        echo "source venv/bin/activate"
        echo
    fi
}

# Function to check and create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    # Create logs directory
    if [ ! -d "logs" ]; then
        mkdir -p logs
        print_success "Created logs directory"
    else
        print_status "logs directory already exists"
    fi
    
    # Create data directory if it doesn't exist
    if [ ! -d "data" ]; then
        mkdir -p data
        print_success "Created data directory"
    else
        print_status "data directory already exists"
    fi
}

# Function to check for required data files
check_data_files() {
    print_status "Checking for sample data files..."
    
    # Check hallucination data files
    if [ -f "hallucination/healthbench_131_negative_cases.json" ]; then
        print_success "Found healthbench_131_negative_cases.json"
    else
        print_warning "healthbench_131_negative_cases.json not found in hallucination/"
    fi
    
    if [ -f "hallucination/stanford_redteaming_129_positive_cases.json" ]; then
        print_success "Found stanford_redteaming_129_positive_cases.json"
    else
        print_warning "stanford_redteaming_129_positive_cases.json not found in hallucination/"
    fi
}

# Function to validate API keys setup
validate_api_keys() {
    print_status "Validating API keys setup..."
    
    if [ -f "config.py" ]; then
        print_success "config.py found"
        
        # Check if API keys are set (basic check)
        if grep -q "your-.*-api-key-here" config.py; then
            print_warning "API keys appear to be using placeholder values"
            print_status "Please run 'python setup_api_keys.py' to configure your API keys"
        else
            print_success "API keys appear to be configured"
        fi
    else
        print_error "config.py not found. Please ensure the file exists."
    fi
}

# Function to run API key setup
run_api_key_setup() {
    print_status "Running API key setup..."
    
    if [ -f "setup_api_keys.py" ]; then
        print_status "Starting interactive API key setup..."
        python3 setup_api_keys.py
    else
        print_error "setup_api_keys.py not found"
    fi
}

# Function to test the installation
test_installation() {
    print_status "Testing installation..."
    
    # Test Python imports
    python3 -c "
import pandas as pd
import pydantic
import openai
import anthropic
from google import genai
from tqdm import tqdm
print('✅ All core packages imported successfully')
" 2>/dev/null && print_success "Core packages test passed" || print_error "Core packages test failed"
    
    # Test config import
    python3 -c "
from config import setup_api_keys, validate_api_keys
print('✅ Config module imported successfully')
" 2>/dev/null && print_success "Config module test passed" || print_error "Config module test failed"
}

# Function to display next steps
display_next_steps() {
    echo
    echo "=========================================="
    echo "🎉 Environment Setup Complete!"
    echo "=========================================="
    echo
    echo "Next steps:"
    echo "1. Configure your API keys:"
    echo "   python setup_api_keys.py"
    echo
    echo "2. Test the installation:"
    echo "   python setup_api_keys.py validate"
    echo
    echo "3. Run your first test:"
    echo "   cd bias && python bias_test.py --help"
    echo "   cd privacy && python privacy_phi_test.py --help"
    echo "   cd hallucination && python agents_v5.py --help"
    echo "   cd robustness && python orchestrator_attacker.py --help"
    echo
    echo "4. Read the README.md for detailed usage instructions"
    echo
    echo "Happy testing! 🧪"
}

# Main setup function
main() {
    echo "=========================================="
    echo "🔬 Medical Red-Teaming Framework Setup"
    echo "=========================================="
    echo
    
    # Check Python version
    if ! check_python_version; then
        print_error "Python 3.10+ is required for this project."
        print_error "Please upgrade your Python installation and try again."
        exit 1
    fi
    
    # Check pip
    if ! check_pip; then
        exit 1
    fi
    
    # Create virtual environment (optional)
    create_virtual_environment
    
    # Install Python packages
    install_python_packages
    
    # Create necessary directories
    create_directories
    
    # Check data files
    check_data_files
    
    # Validate API keys setup
    validate_api_keys
    
    # Test installation
    test_installation
    
    # Display next steps
    display_next_steps
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being executed
    main "$@"
else
    # Script is being sourced
    print_warning "This script should be executed, not sourced"
fi 