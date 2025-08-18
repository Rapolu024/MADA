# MADA Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Docker Desktop 4.0+ installed and running
- 8GB+ RAM available
- 10GB+ disk space

### One-Command Deployment
```bash
# Clone and deploy
git clone <repository>
cd MADA
./scripts/deploy.sh
```

That's it! 🎉 

Access MADA at: **http://localhost:8501**

---

## 📋 What Gets Deployed

### Core Services
- **🏥 MADA Dashboard** (Port 8501) - Main medical interface
- **🔧 API Server** (Port 8000) - REST API for integrations
- **💾 PostgreSQL** (Port 5432) - HIPAA-compliant database
- **⚡ Redis** (Port 6379) - Caching and session management

### Monitoring Stack
- **📈 Grafana** (Port 3000) - Metrics dashboards
- **🔍 Prometheus** (Port 9090) - Metrics collection
- **📋 Kibana** (Port 5601) - Log analysis
- **🔎 Elasticsearch** (Port 9200) - Log storage

### Infrastructure
- **🌐 Nginx** (Port 80) - Load balancer and reverse proxy
- **📊 Logstash** - Log processing pipeline

---

## 🔧 Manual Deployment Steps

If you prefer step-by-step deployment:

### 1. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit with your settings
nano .env
```

### 2. Build and Start
```bash
# Build all images
docker-compose build

# Start services
docker-compose up -d

# Check status
docker-compose ps
```

### 3. Initialize Database
```bash
# Run database migrations (when available)
docker-compose exec mada_api python manage.py migrate

# Create initial data
docker-compose exec mada_api python scripts/create_sample_data.py
```

---

## 📊 Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| **MADA Dashboard** | http://localhost:8501 | N/A |
| **API Documentation** | http://localhost:8000/docs | N/A |
| **Grafana** | http://localhost:3000 | admin / (see .env) |
| **Prometheus** | http://localhost:9090 | N/A |
| **Kibana** | http://localhost:5601 | N/A |

---

## 🛠️ Management Commands

### Service Control
```bash
# View all services
docker-compose ps

# View logs
docker-compose logs -f

# Restart specific service
docker-compose restart mada_dashboard

# Stop all services
docker-compose down

# Complete cleanup
./scripts/deploy.sh clean
```

### Data Management
```bash
# Backup database
docker-compose exec postgres pg_dump -U mada_user mada_db > backup.sql

# Restore database
docker-compose exec -i postgres psql -U mada_user mada_db < backup.sql

# View database
docker-compose exec postgres psql -U mada_user mada_db
```

### System Monitoring
```bash
# Check resource usage
docker stats

# Monitor logs in real-time
docker-compose logs -f mada_dashboard

# System health check
curl http://localhost:8501/_stcore/health
```

---

## 🔒 Security Configuration

### Production Checklist
- [ ] Update default passwords in `.env`
- [ ] Enable SSL certificates
- [ ] Configure firewall rules
- [ ] Setup backup automation
- [ ] Enable audit logging
- [ ] Review HIPAA compliance settings

### SSL Setup (Production)
```bash
# Generate SSL certificates
mkdir -p nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/privkey.pem \
  -out nginx/ssl/fullchain.pem

# Update docker-compose.yml to use HTTPS
# Update nginx.conf for SSL configuration
```

---

## 📈 Performance Tuning

### Resource Allocation
```yaml
# In docker-compose.yml
services:
  mada_dashboard:
    deploy:
      resources:
        limits:
          memory: 2g
          cpus: '1.0'
        reservations:
          memory: 1g
          cpus: '0.5'
```

### Database Optimization
```sql
-- Optimize PostgreSQL for medical workloads
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
SELECT pg_reload_conf();
```

---

## 🧪 Testing the Deployment

### Automated Health Checks
```bash
# Run health check script
./scripts/health_check.sh

# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/api/patients

# Test dashboard connectivity
curl http://localhost:8501/_stcore/health
```

### Manual Testing Steps
1. **Access Dashboard**: Navigate to http://localhost:8501
2. **Create Patient**: Use "Patient Management" → "New Patient Intake"
3. **Generate Diagnosis**: Use "AI Diagnosis" with sample patient
4. **View Analytics**: Check "Analytics & Reports" section
5. **Monitor System**: Access Grafana at http://localhost:3000

---

## 🔧 Troubleshooting

### Common Issues

#### Services Won't Start
```bash
# Check Docker resource limits
docker system df
docker system prune

# Verify ports are available
sudo lsof -i :8501
sudo lsof -i :5432
```

#### Database Connection Errors
```bash
# Check PostgreSQL status
docker-compose logs postgres

# Verify credentials
docker-compose exec postgres psql -U mada_user -d mada_db -c "\l"

# Reset database
docker-compose down -v
docker-compose up postgres -d
```

#### Dashboard Loading Issues
```bash
# Check Streamlit logs
docker-compose logs mada_dashboard

# Verify Python dependencies
docker-compose exec mada_dashboard pip list

# Clear cache
docker-compose restart mada_dashboard
```

#### Memory Issues
```bash
# Increase Docker memory limit to 8GB+
# Restart services one by one
docker-compose restart postgres
docker-compose restart mada_dashboard
```

### Performance Issues
```bash
# Monitor resource usage
docker stats

# Check database performance
docker-compose exec postgres pg_stat_database

# Optimize containers
docker-compose down
docker system prune
docker-compose up -d
```

---

## 📊 Monitoring and Alerting

### Grafana Dashboards
- **System Health**: CPU, Memory, Disk usage
- **Application Metrics**: Request rates, response times
- **Medical Metrics**: Patient volume, diagnosis accuracy
- **Database Performance**: Query performance, connections

### Log Analysis with Kibana
- **Application Logs**: MADA dashboard and API logs
- **System Logs**: Container and infrastructure logs
- **Audit Logs**: HIPAA compliance tracking
- **Error Analysis**: Exception tracking and trends

### Custom Alerts
```yaml
# Example Prometheus alert rules
groups:
  - name: mada-alerts
    rules:
      - alert: HighMemoryUsage
        expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
```

---

## 🔄 Updates and Maintenance

### Regular Updates
```bash
# Update to latest images
docker-compose pull
docker-compose up -d

# Update MADA application
git pull origin main
docker-compose build --no-cache
docker-compose up -d
```

### Backup Strategy
```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
docker-compose exec postgres pg_dump -U mada_user mada_db > backup_${DATE}.sql
tar -czf mada_backup_${DATE}.tar.gz backup_${DATE}.sql data/
```

### Maintenance Windows
1. **Weekly**: Update system packages and Docker images
2. **Monthly**: Database optimization and cleanup
3. **Quarterly**: Security audit and compliance review
4. **Annually**: Full system backup and disaster recovery test

---

## 🆘 Support and Documentation

### Getting Help
- 📚 **Documentation**: [Link to full docs]
- 🐛 **Bug Reports**: [Link to issue tracker]
- 💬 **Community**: [Link to discussions]
- 📧 **Support**: [Contact information]

### Useful Resources
- Docker Compose Reference: https://docs.docker.com/compose/
- PostgreSQL Documentation: https://www.postgresql.org/docs/
- Streamlit Documentation: https://docs.streamlit.io/
- HIPAA Compliance Guide: [Internal compliance documentation]

---

## ✅ Production Readiness Checklist

### Security ✅
- [ ] SSL/TLS certificates configured
- [ ] Strong passwords and secrets rotation
- [ ] Network security and firewall rules
- [ ] Audit logging enabled
- [ ] HIPAA compliance review completed

### Performance ✅
- [ ] Resource limits configured
- [ ] Database tuned for workload
- [ ] Monitoring and alerting setup
- [ ] Load testing completed
- [ ] Backup and recovery tested

### Operations ✅
- [ ] Health checks implemented
- [ ] Log aggregation configured
- [ ] Automated deployment pipeline
- [ ] Documentation updated
- [ ] Staff training completed

---

## 🎉 Success!

Your MADA system is now deployed and ready for medical AI diagnosis assistance!

**Next Steps:**
1. Train AI models with your medical data
2. Configure clinic-specific settings
3. Setup user accounts and permissions
4. Begin clinical validation testing
5. Scale according to patient volume

**Questions?** Contact the MADA support team or check our comprehensive documentation.
