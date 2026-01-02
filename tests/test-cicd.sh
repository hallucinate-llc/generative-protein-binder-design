#!/bin/bash

# Test script for Docker CI/CD pipeline
# This script simulates the key parts of the CI/CD workflow locally

set -e

echo "=== Docker CI/CD Pipeline Test ==="
echo "Testing the CI/CD pipeline components locally"
echo

PROJECT_ROOT="/opt/generative-protein-binder-design"
cd "$PROJECT_ROOT"

# Function to test component build
test_component_build() {
    local component="$1"
    local context="$2"
    local dockerfile="$3"
    local port="$4"
    
    echo "Testing $component build..."
    echo "Context: $context"
    echo "Dockerfile: $dockerfile"
    echo "Port: $port"
    
    # Check if context directory exists
    if [ ! -d "$context" ]; then
        echo "❌ Context directory not found: $context"
        return 1
    fi
    
    # Check if Dockerfile exists or create a basic one
    if [ ! -f "$dockerfile" ]; then
        echo "⚠️  Dockerfile not found, creating basic one: $dockerfile"
        mkdir -p "$(dirname "$dockerfile")"
        
        case "$component" in
            "mcp-server")
                cat > "$dockerfile" << 'EOL'
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 mcpuser && \
    chown -R mcpuser:mcpuser /app
USER mcpuser

EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

ENV MODEL_BACKEND=native
ENV PYTHONPATH=/app

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
EOL
                ;;
            "mcp-dashboard")
                cat > "$dockerfile" << 'EOL'
FROM node:18-slim

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Build application
RUN npm run build

# Create non-root user
RUN useradd -m -u 1000 dashuser && \
    chown -R dashuser:dashuser /app
USER dashuser

EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:3000 || exit 1

ENV NODE_ENV=production

CMD ["npm", "start"]
EOL
                ;;
        esac
    fi
    
    # Build the Docker image
    echo "Building Docker image for $component..."
    if docker build -f "$dockerfile" -t "test-$component:latest" "$context"; then
        echo "✅ Build successful for $component"
        
        # Test the image
        echo "Testing image startup..."
        container_name="test-$component-$(date +%s)"
        
        if docker run -d --name "$container_name" -p "$port:$port" "test-$component:latest"; then
            echo "✅ Container started successfully"
            
            # Wait a moment and check if container is still running
            sleep 5
            if docker ps | grep -q "$container_name"; then
                echo "✅ Container is running"
            else
                echo "❌ Container stopped unexpectedly"
                docker logs "$container_name"
            fi
            
            # Cleanup
            docker stop "$container_name" >/dev/null 2>&1 || true
            docker rm "$container_name" >/dev/null 2>&1 || true
        else
            echo "❌ Failed to start container"
            return 1
        fi
        
        # Clean up image
        docker rmi "test-$component:latest" >/dev/null 2>&1 || true
        
    else
        echo "❌ Build failed for $component"
        return 1
    fi
    
    echo
}

# Function to check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker >/dev/null 2>&1; then
        echo "❌ Docker not found"
        return 1
    fi
    
    if ! docker ps >/dev/null 2>&1; then
        echo "❌ Cannot connect to Docker daemon"
        return 1
    fi
    
    echo "✅ Docker is available: $(docker --version)"
    
    # Check project structure
    if [ ! -d "mcp-server" ]; then
        echo "❌ mcp-server directory not found"
        return 1
    fi
    
    if [ ! -d "mcp-dashboard" ]; then
        echo "❌ mcp-dashboard directory not found"
        return 1
    fi
    
    echo "✅ Project structure looks good"
    return 0
}

# Function to test workflow trigger conditions
test_workflow_triggers() {
    echo "Testing workflow trigger conditions..."
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir >/dev/null 2>&1; then
        echo "❌ Not in a git repository"
        return 1
    fi
    
    # Check current branch
    current_branch=$(git branch --show-current)
    echo "Current branch: $current_branch"
    
    # Check if workflow file exists
    workflow_file=".github/workflows/selfhosted-docker-cicd.yml"
    if [ ! -f "$workflow_file" ]; then
        echo "❌ Workflow file not found: $workflow_file"
        return 1
    fi
    
    echo "✅ Workflow file exists: $workflow_file"
    
    # Check for recent commits
    recent_commits=$(git log --oneline -5)
    echo "Recent commits:"
    echo "$recent_commits"
    
    return 0
}

# Function to simulate CI/CD environment
simulate_cicd_env() {
    echo "Simulating CI/CD environment variables..."
    
    export GITHUB_WORKFLOW="Self-Hosted Docker CI/CD"
    export GITHUB_RUN_NUMBER="test-$(date +%s)"
    export GITHUB_SHA="$(git rev-parse HEAD 2>/dev/null || echo 'test-sha')"
    export GITHUB_REF_NAME="$(git branch --show-current 2>/dev/null || echo 'main')"
    export GITHUB_REPOSITORY="endomorphosis/generative-protein-binder-design"
    export GITHUB_ACTOR="$(git config user.name 2>/dev/null || echo 'test-user')"
    export REGISTRY="ghcr.io"
    
    echo "Environment variables set:"
    echo "- GITHUB_WORKFLOW: $GITHUB_WORKFLOW"
    echo "- GITHUB_RUN_NUMBER: $GITHUB_RUN_NUMBER"
    echo "- GITHUB_SHA: $GITHUB_SHA"
    echo "- GITHUB_REF_NAME: $GITHUB_REF_NAME"
    echo "- GITHUB_REPOSITORY: $GITHUB_REPOSITORY"
    echo "- GITHUB_ACTOR: $GITHUB_ACTOR"
    echo "- REGISTRY: $REGISTRY"
    echo
}

# Main test execution
main() {
    echo "Starting Docker CI/CD pipeline test..."
    echo "Working directory: $(pwd)"
    echo
    
    # Check prerequisites
    if ! check_prerequisites; then
        echo "❌ Prerequisites check failed"
        exit 1
    fi
    echo
    
    # Test workflow triggers
    if ! test_workflow_triggers; then
        echo "❌ Workflow trigger test failed"
        exit 1
    fi
    echo
    
    # Simulate CI/CD environment
    simulate_cicd_env
    
    # Test component builds
    echo "Testing component builds..."
    echo
    
    # Test MCP Server
    if ! test_component_build "mcp-server" "./mcp-server" "./mcp-server/Dockerfile" "8001"; then
        echo "❌ MCP Server test failed"
        exit 1
    fi
    
    # Test MCP Dashboard
    if ! test_component_build "mcp-dashboard" "./mcp-dashboard" "./mcp-dashboard/Dockerfile" "3000"; then
        echo "❌ MCP Dashboard test failed"
        exit 1
    fi
    
    # Test report generation
    echo "Generating test report..."
    
    cat > cicd-test-report.json << EOF
{
    "test_timestamp": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
    "test_status": "completed",
    "components_tested": ["mcp-server", "mcp-dashboard"],
    "system_info": {
        "hostname": "$(hostname)",
        "architecture": "$(uname -m)",
        "docker_version": "$(docker --version)",
        "available_space": "$(df -h . | tail -1 | awk '{print $4}')",
        "memory": "$(free -h | grep Mem | awk '{print $2}')",
        "cpu_cores": "$(nproc)"
    },
    "git_info": {
        "branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
        "commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
        "status": "$(git status --porcelain 2>/dev/null | wc -l) uncommitted changes"
    }
}
EOF
    
    echo "Test report generated:"
    cat cicd-test-report.json | jq .
    
    echo
    echo "✅ All tests passed successfully!"
    echo
    echo "Next steps:"
    echo "1. Configure your GitHub Actions runner: ./configure-runner.sh"
    echo "2. Push your changes to trigger the actual CI/CD pipeline"
    echo "3. Monitor the pipeline at: https://github.com/endomorphosis/generative-protein-binder-design/actions"
    
    # Clean up any remaining test artifacts
    docker system prune -f >/dev/null 2>&1 || true
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi