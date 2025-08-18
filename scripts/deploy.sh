#!/bin/bash

# MADA Deployment Script
# Comprehensive deployment for Medical AI Diagnosis Assistant

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="mada"
DOCKER_COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

echo -e "${BLUE}🏥 MADA - Medical AI Diagnosis Assistant${NC}"
echo -e "${BLUE}=======================================${NC}"
echo ""

# Function to print colored output
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed and running
check_docker() {
    log_info "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    log_success "Docker and Docker Compose are ready"
}

# Create environment file if it doesn't exist
setup_environment() {
    log_info "Setting up environment configuration..."
    
    if [ ! -f "$ENV_FILE" ]; then
        log_info "Creating environment file from template..."
        
        # Generate secure random passwords
        DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
        SECRET_KEY=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
        GRAFANA_PASSWORD=$(openssl rand -base64 16 | tr -d "=+/" | cut -c1-12)
        
        cat > $ENV_FILE << EOL
# MADA Environment Configuration
# Generated on $(date)

# Database
DB_PASSWORD=${DB_PASSWORD}

# Application Security
SECRET_KEY=${SECRET_KEY}

# Monitoring
GRAFANA_PASSWORD=${GRAFANA_PASSWORD}

# Training Schedule (cron format)
TRAINING_SCHEDULE=0 2 * * *

# Environment
ENVIRONMENT=production
DEBUG=False

# Logging
LOG_LEVEL=INFO
EOL
        
        log_success "Environment file created with secure random passwords"
        log_warning "Please review and update $ENV_FILE as needed"
    else
        log_info "Environment file already exists"
    fi
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p data/{raw,processed,models}
    mkdir -p logs
    mkdir -p nginx/ssl
    mkdir -p monitoring/{prometheus,grafana/datasources,grafana/dashboards,logstash}
    mkdir -p scripts
    
    log_success "Directories created"
}

# Setup nginx configuration
setup_nginx() {
    log_info "Setting up Nginx configuration..."
    
    cat > nginx/nginx.conf << EOL
events {
    worker_connections 1024;
}

http {
    upstream mada_dashboard {
        server mada_dashboard:8501;
    }
    
    upstream mada_api {
        server mada_api:8000;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        # Health check endpoint
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
        
        # Dashboard
        location / {
            proxy_pass http://mada_dashboard;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            
            # WebSocket support for Streamlit
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection "upgrade";
        }
        
        # API endpoints
        location /api/ {
            proxy_pass http://mada_api/;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
    }
}
EOL
    
    log_success "Nginx configuration created"
}

# Setup monitoring configuration
setup_monitoring() {
    log_info "Setting up monitoring configuration..."
    
    # Prometheus configuration
    cat > monitoring/prometheus.yml << EOL
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'mada-dashboard'
    static_configs:
      - targets: ['mada_dashboard:8501']
  
  - job_name: 'mada-api'
    static_configs:
      - targets: ['mada_api:8000']
  
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
  
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
EOL
    
    # Logstash configuration
    cat > monitoring/logstash/logstash.conf << EOL
input {
  file {
    path => "/app/logs/*.log"
    start_position => "beginning"
  }
}

filter {
  if [path] =~ /\.log$/ {
    grok {
      match => { "message" => "%{TIMESTAMP_ISO8601:timestamp} - %{WORD:logger} - %{LOGLEVEL:level} - %{GREEDYDATA:message}" }
    }
    date {
      match => [ "timestamp", "yyyy-MM-dd HH:mm:ss,SSS" ]
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "mada-logs-%{+YYYY.MM.dd}"
  }
}
EOL
    
    log_success "Monitoring configuration created"
}

# Setup database initialization script
setup_database() {
    log_info "Setting up database initialization..."
    
    cat > scripts/init_db.sql << EOL
-- MADA Database Initialization
-- Create extensions and initial setup

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create application-specific schemas
CREATE SCHEMA IF NOT EXISTS mada;
CREATE SCHEMA IF NOT EXISTS audit;

-- Set default privileges
GRANT USAGE ON SCHEMA mada TO mada_user;
GRANT CREATE ON SCHEMA mada TO mada_user;
GRANT USAGE ON SCHEMA audit TO mada_user;
GRANT CREATE ON SCHEMA audit TO mada_user;

-- Create audit function for automatic logging
CREATE OR REPLACE FUNCTION audit.log_changes()
RETURNS TRIGGER AS \$\$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit.activity_log (table_name, action, new_data, created_at)
        VALUES (TG_TABLE_NAME, 'INSERT', row_to_json(NEW), NOW());
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit.activity_log (table_name, action, old_data, new_data, created_at)
        VALUES (TG_TABLE_NAME, 'UPDATE', row_to_json(OLD), row_to_json(NEW), NOW());
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit.activity_log (table_name, action, old_data, created_at)
        VALUES (TG_TABLE_NAME, 'DELETE', row_to_json(OLD), NOW());
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
\$\$ LANGUAGE plpgsql;

-- Create initial audit log table
CREATE TABLE IF NOT EXISTS audit.activity_log (
    id SERIAL PRIMARY KEY,
    table_name TEXT NOT NULL,
    action TEXT NOT NULL,
    old_data JSONB,
    new_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON SCHEMA mada IS 'MADA application data';
COMMENT ON SCHEMA audit IS 'Audit logging for HIPAA compliance';
EOL
    
    log_success "Database initialization script created"
}

# Build and deploy the application
deploy_application() {
    log_info "Building and deploying MADA application..."
    
    # Stop any existing containers
    docker-compose down 2>/dev/null || true
    
    # Build the application
    log_info "Building Docker images..."
    docker-compose build --no-cache
    
    # Start the services
    log_info "Starting services..."
    docker-compose up -d
    
    log_success "Application deployed successfully"
}

# Health check for deployed services
health_check() {
    log_info "Performing health checks..."
    
    # Wait for services to be ready
    sleep 30
    
    services=("postgres:5432" "redis:6379" "mada_dashboard:8501")
    
    for service in "${services[@]}"; do
        container_name=$(echo $service | cut -d: -f1)
        
        if docker-compose ps $container_name | grep -q "Up"; then
            log_success "$container_name is running"
        else
            log_error "$container_name is not running properly"
        fi
    done
    
    # Check web accessibility
    if curl -f -s http://localhost:8501/_stcore/health > /dev/null; then
        log_success "Dashboard is accessible at http://localhost:8501"
    else
        log_warning "Dashboard health check failed - may still be starting up"
    fi
}

# Display deployment information
show_deployment_info() {
    echo ""
    echo -e "${GREEN}🎉 MADA Deployment Complete!${NC}"
    echo -e "${GREEN}=========================${NC}"
    echo ""
    echo -e "${BLUE}Access URLs:${NC}"
    echo "  📊 Dashboard:     http://localhost:8501"
    echo "  🔧 API:          http://localhost:8000"
    echo "  📈 Grafana:      http://localhost:3000"
    echo "  🔍 Prometheus:   http://localhost:9090"
    echo "  📋 Kibana:       http://localhost:5601"
    echo ""
    echo -e "${BLUE}Default Credentials:${NC}"
    echo "  Grafana: admin / $(grep GRAFANA_PASSWORD $ENV_FILE | cut -d= -f2)"
    echo ""
    echo -e "${BLUE}Useful Commands:${NC}"
    echo "  View logs:       docker-compose logs -f"
    echo "  Stop services:   docker-compose down"
    echo "  Restart:         docker-compose restart"
    echo "  Update:          docker-compose pull && docker-compose up -d"
    echo ""
    echo -e "${YELLOW}Note: First startup may take 2-3 minutes for all services to be ready${NC}"
}

# Main deployment process
main() {
    log_info "Starting MADA deployment process..."
    
    check_docker
    setup_environment
    create_directories
    setup_nginx
    setup_monitoring
    setup_database
    deploy_application
    health_check
    show_deployment_info
    
    log_success "MADA deployment completed successfully!"
}

# Handle script arguments
case "${1:-}" in
    "clean")
        log_warning "Cleaning up MADA deployment..."
        docker-compose down -v
        docker system prune -f
        log_success "Cleanup completed"
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "status")
        docker-compose ps
        ;;
    "restart")
        docker-compose restart
        ;;
    *)
        main
        ;;
esac
