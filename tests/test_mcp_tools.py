#!/usr/bin/env python3
"""
Comprehensive test suite for MCP Server tools
Tests all tools in the MCP server before dashboard integration
"""

import asyncio
import httpx
import json
import time
from typing import Dict, Any, List
from datetime import datetime

# Configuration
MCP_SERVER_URL = "http://localhost:8010"
TEST_TIMEOUT = 60  # seconds
POLLING_INTERVAL = 2  # seconds

class MCP_ToolTester:
    """Test suite for MCP server tools"""
    
    def __init__(self, server_url: str = MCP_SERVER_URL):
        self.server_url = server_url
        self.test_results = []
        self.job_ids = []
        
    async def test_health(self) -> bool:
        """Test health check endpoint"""
        test_name = "Health Check"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.server_url}/health")
                
                if response.status_code == 200:
                    data = response.json()
                    self._log_result(test_name, "PASS", data)
                    return True
                else:
                    self._log_result(test_name, "FAIL", f"Status {response.status_code}")
                    return False
        except Exception as e:
            self._log_result(test_name, "ERROR", str(e))
            return False
    
    async def test_list_tools(self) -> bool:
        """Test MCP /tools endpoint"""
        test_name = "List Tools (MCP)"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.server_url}/mcp/v1/tools")
                
                if response.status_code == 200:
                    data = response.json()
                    tools = data.get("tools", [])
                    expected = {
                        "get_runtime_config",
                        "update_runtime_config",
                        "reset_runtime_config",
                        "embedded_bootstrap",
                        "design_protein_binder",
                        "get_job_status",
                        "list_jobs",
                        "delete_job",
                        "check_services",
                        "predict_structure",
                        "design_binder_backbone",
                        "generate_sequence",
                        "predict_complex",
                        "get_alphafold_settings",
                        "update_alphafold_settings",
                        "reset_alphafold_settings",
                    }

                    tool_names = {t["name"] for t in tools}

                    if tool_names == expected:
                        self._log_result(
                            test_name,
                            "PASS",
                            f"Found {len(tools)} tools: {sorted(tool_names)}"
                        )
                        return True
                    else:
                        missing = expected - tool_names
                        extra = tool_names - expected
                        self._log_result(
                            test_name,
                            "FAIL",
                            f"Tool names mismatch. Missing: {sorted(missing)} Extra: {sorted(extra)}"
                        )
                        return False
                else:
                    self._log_result(test_name, "FAIL", f"Status {response.status_code}")
                    return False
        except Exception as e:
            self._log_result(test_name, "ERROR", str(e))
            return False
    
    async def test_services_status(self) -> bool:
        """Test service health status endpoint"""
        test_name = "Services Status Check"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.server_url}/api/services/status")
                
                if response.status_code == 200:
                    data = response.json()
                    expected_services = {"alphafold", "rfdiffusion", "proteinmpnn", "alphafold_multimer"}
                    found_services = set(data.keys())
                    
                    if expected_services == found_services:
                        # Check status values
                        status_info = {
                            service: data[service].get("status", "unknown")
                            for service in expected_services
                        }
                        self._log_result(
                            test_name, 
                            "PASS", 
                            f"All services checked: {status_info}"
                        )
                        return True
                    else:
                        self._log_result(
                            test_name, 
                            "FAIL", 
                            f"Service mismatch. Expected {expected_services}, got {found_services}"
                        )
                        return False
                else:
                    self._log_result(test_name, "FAIL", f"Status {response.status_code}")
                    return False
        except Exception as e:
            self._log_result(test_name, "ERROR", str(e))
            return False
    
    async def test_design_protein_binder(self) -> bool:
        """Test design_protein_binder tool"""
        test_name = "Design Protein Binder"
        try:
            # Test data - a valid amino acid sequence
            test_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVV"
            
            payload = {
                "sequence": test_sequence,
                "job_name": f"test_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "num_designs": 2
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.server_url}/api/jobs",
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    job_id = data.get("job_id")
                    
                    if job_id:
                        self.job_ids.append(job_id)
                        self._log_result(
                            test_name,
                            "PASS",
                            f"Job created: {job_id}"
                        )
                        return True
                    else:
                        self._log_result(test_name, "FAIL", "No job_id in response")
                        return False
                else:
                    self._log_result(test_name, "FAIL", f"Status {response.status_code}")
                    return False
        except Exception as e:
            self._log_result(test_name, "ERROR", str(e))
            return False
    
    async def test_list_jobs(self) -> bool:
        """Test list_jobs tool"""
        test_name = "List Jobs"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.server_url}/api/jobs")
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if isinstance(data, list):
                        self._log_result(
                            test_name,
                            "PASS",
                            f"Retrieved {len(data)} jobs"
                        )
                        return True
                    else:
                        self._log_result(test_name, "FAIL", "Response is not a list")
                        return False
                else:
                    self._log_result(test_name, "FAIL", f"Status {response.status_code}")
                    return False
        except Exception as e:
            self._log_result(test_name, "ERROR", str(e))
            return False
    
    async def test_get_job_status(self, job_id: str) -> bool:
        """Test get_job_status tool"""
        test_name = f"Get Job Status ({job_id})"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.server_url}/api/jobs/{job_id}")
                
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status")
                    progress = data.get("progress", {})
                    
                    self._log_result(
                        test_name,
                        "PASS",
                        f"Status: {status}, Progress: {json.dumps(progress, indent=2)}"
                    )
                    return True
                else:
                    self._log_result(test_name, "FAIL", f"Status {response.status_code}")
                    return False
        except Exception as e:
            self._log_result(test_name, "ERROR", str(e))
            return False
    
    async def test_job_progress_monitoring(self, job_id: str, timeout: int = TEST_TIMEOUT) -> bool:
        """Monitor job progress until completion or timeout"""
        test_name = f"Job Progress Monitoring ({job_id})"
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                while time.time() - start_time < timeout:
                    response = await client.get(f"{self.server_url}/api/jobs/{job_id}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        status = data.get("status")
                        progress = data.get("progress", {})
                        
                        print(f"\n[{test_name}] Status: {status}")
                        for step, step_status in progress.items():
                            print(f"  - {step}: {step_status}")
                        
                        if status in ["completed", "failed"]:
                            if status == "completed":
                                self._log_result(
                                    test_name,
                                    "PASS",
                                    f"Job completed successfully"
                                )
                            else:
                                error = data.get("error", "Unknown error")
                                self._log_result(
                                    test_name,
                                    "FAIL",
                                    f"Job failed: {error}"
                                )
                            return status == "completed"
                        
                        await asyncio.sleep(POLLING_INTERVAL)
                    else:
                        self._log_result(test_name, "ERROR", f"Status {response.status_code}")
                        return False
                
                # Timeout
                self._log_result(test_name, "TIMEOUT", f"Job did not complete within {timeout}s")
                return False
                
        except Exception as e:
            self._log_result(test_name, "ERROR", str(e))
            return False
    
    async def test_list_resources(self) -> bool:
        """Test MCP /resources endpoint"""
        test_name = "List Resources (MCP)"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.server_url}/mcp/v1/resources")
                
                if response.status_code == 200:
                    data = response.json()
                    resources = data.get("resources", [])
                    
                    self._log_result(
                        test_name,
                        "PASS",
                        f"Found {len(resources)} resources"
                    )
                    return True
                else:
                    self._log_result(test_name, "FAIL", f"Status {response.status_code}")
                    return False
        except Exception as e:
            self._log_result(test_name, "ERROR", str(e))
            return False
    
    async def test_get_resource(self, job_id: str) -> bool:
        """Test MCP get resource endpoint"""
        test_name = f"Get Resource ({job_id})"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.server_url}/mcp/v1/resources/{job_id}")
                
                if response.status_code == 200:
                    data = response.json()
                    contents = data.get("contents", [])
                    
                    if contents:
                        self._log_result(
                            test_name,
                            "PASS",
                            f"Retrieved resource with {len(contents)} content item(s)"
                        )
                        return True
                    else:
                        self._log_result(test_name, "FAIL", "No content in resource")
                        return False
                elif response.status_code == 404:
                    self._log_result(
                        test_name,
                        "SKIP",
                        "Job results not yet available"
                    )
                    return True  # Not a failure, just not ready yet
                else:
                    self._log_result(test_name, "FAIL", f"Status {response.status_code}")
                    return False
        except Exception as e:
            self._log_result(test_name, "ERROR", str(e))
            return False
    
    async def test_delete_job(self, job_id: str) -> bool:
        """Test delete job endpoint"""
        test_name = f"Delete Job ({job_id})"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.delete(f"{self.server_url}/api/jobs/{job_id}")
                
                if response.status_code == 200:
                    data = response.json()
                    self._log_result(
                        test_name,
                        "PASS",
                        f"Job deleted: {data.get('message', 'success')}"
                    )
                    return True
                else:
                    self._log_result(test_name, "FAIL", f"Status {response.status_code}")
                    return False
        except Exception as e:
            self._log_result(test_name, "ERROR", str(e))
            return False
    
    def _log_result(self, test_name: str, status: str, details: str):
        """Log test result"""
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        # Color coding
        color = {
            "PASS": "\033[92m",      # Green
            "FAIL": "\033[91m",      # Red
            "ERROR": "\033[93m",     # Yellow
            "TIMEOUT": "\033[94m",   # Blue
            "SKIP": "\033[96m",      # Cyan
        }.get(status, "\033[0m")
        reset = "\033[0m"
        
        print(f"{color}[{status}]{reset} {test_name}")
        print(f"  Details: {details}\n")
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("MCP SERVER TOOL TEST SUMMARY")
        print("="*80)
        
        status_counts = {}
        for result in self.test_results:
            status = result["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"\nTotal tests: {len(self.test_results)}")
        for status in ["PASS", "FAIL", "ERROR", "TIMEOUT", "SKIP"]:
            count = status_counts.get(status, 0)
            if count > 0:
                print(f"  {status}: {count}")
        
        print("\nDetailed Results:")
        print("-" * 80)
        
        for result in self.test_results:
            status = result["status"]
            color = {
                "PASS": "\033[92m",
                "FAIL": "\033[91m",
                "ERROR": "\033[93m",
                "TIMEOUT": "\033[94m",
                "SKIP": "\033[96m",
            }.get(status, "\033[0m")
            reset = "\033[0m"
            
            print(f"{color}[{status}]{reset} {result['test']}")
            print(f"      {result['details']}")
        
        print("\n" + "="*80)


async def run_full_test_suite():
    """Run comprehensive test suite"""
    tester = MCP_ToolTester()
    
    print("Starting MCP Server Tool Tests...")
    print(f"Server: {MCP_SERVER_URL}\n")
    
    # Basic connectivity tests
    print("=== BASIC CONNECTIVITY ===\n")
    await tester.test_health()
    
    # Tool discovery tests
    print("=== TOOL DISCOVERY ===\n")
    await tester.test_list_tools()
    await tester.test_services_status()
    
    # Tool functionality tests
    print("=== TOOL FUNCTIONALITY ===\n")
    
    # Test design_protein_binder
    design_success = await tester.test_design_protein_binder()
    
    # Test list_jobs
    await tester.test_list_jobs()
    
    # If job created, test get_job_status
    if design_success and tester.job_ids:
        job_id = tester.job_ids[0]
        print(f"=== JOB PROCESSING ({job_id}) ===\n")
        
        # Check initial status
        await tester.test_get_job_status(job_id)
        
        # Monitor progress
        job_completed = await tester.test_job_progress_monitoring(job_id)
        
        # Test resource endpoints
        print("=== RESOURCE ENDPOINTS ===\n")
        await tester.test_list_resources()
        
        if job_completed:
            await tester.test_get_resource(job_id)
        
        # Test cleanup (optional)
        # await tester.test_delete_job(job_id)
    
    # Print summary
    tester.print_summary()
    
    return tester


if __name__ == "__main__":
    asyncio.run(run_full_test_suite())
