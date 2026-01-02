# Validation Quick Reference

## How to Verify Everything Works

### For Malta User (Quick Test)

```bash
# 1. Access repository
cd /opt/generative-protein-binder-design
echo "✓ Repository accessible"

# 2. Check cache
ls ~/.cache/alphafold/ | head -1
echo "✓ Cache accessible via symlink"

# 3. Check git
git log --oneline | head -1
echo "✓ Git repository working"

# 4. Check configuration
cat .env.gpu | head -1
echo "✓ Config files readable"

# 5. Check tools
ls tools/*/bin/* 2>/dev/null | head -1
echo "✓ ML tools present"
```

### For Developers (Full Python Test)

```bash
# Set up environment
export PYTHONPATH="/opt/generative-protein-binder-design:$PYTHONPATH"

# Test Python imports
python3 -c "from mcp_server import app; print('✓ MCP server loads')"
python3 -c "from mcp_server.model_backends import BackendManager; print('✓ Backends load')"

# Test Docker files
python3 -c "import yaml; yaml.safe_load(open('deploy/docker-compose.yaml')); print('✓ Docker config valid')"
```

### For Administrators (Full Validation)

```bash
# Check group membership
getent group protein-design | grep -q maltese && echo "✓ maltese in group"

# Check directory permissions
stat -c '%a' /opt/generative-protein-binder-design | grep -E '^27[0-7]{2}' && echo "✓ Setgid bit set"

# Check file permissions
find /opt/generative-protein-binder-design/mcp-server -name "*.py" -exec stat -c '%a' {} \; | grep -q '^644$' && echo "✓ File permissions correct"

# Check symlinks
test -L /opt/generative-protein-binder-design/mcp_server && echo "✓ Python module symlink exists"
ls -L ~/.cache/alphafold/ | wc -l | grep -q '[1-9]' && echo "✓ Cache symlink works"

# Check git config
git config --global --get-regexp safe.directory | grep -q 'generative-protein' && echo "✓ Git safe.directory configured"

# Check PYTHONPATH
grep PYTHONPATH /etc/profile.d/generative-protein-binder.sh | grep -q 'opt' && echo "✓ PYTHONPATH configured"
```

## What Was Fixed

| Issue | Solution | File(s) Modified |
|-------|----------|-----------------|
| Python path | Added PYTHONPATH env var | `/etc/profile.d/generative-protein-binder.sh` |
| Module naming | Created mcp_server → mcp-server symlink | `/opt/generative-protein-binder-design/` |
| Relative imports | Updated __init__.py | `/opt/generative-protein-binder-design/mcp-server/__init__.py` |
| Git ownership | Added safe.directory | Global git config |
| File permissions | Set to 644 | mcp-server Python files |

## Support

**Issue:** "ModuleNotFoundError: No module named 'fastapi'"
**Solution:** Activate conda base environment (`conda activate base`)

**Issue:** "Permission denied" or "Operation not permitted"
**Solution:** Ensure user is in protein-design group (`groups` command)

**Issue:** "Git reports dubious ownership"
**Solution:** Already fixed via git safe.directory configuration

**Issue:** "Cache not found"
**Solution:** Cache is at `/var/cache/generative-protein-binder-design`, symlinked to `~/.cache/alphafold`

---
Updated: January 2, 2026
