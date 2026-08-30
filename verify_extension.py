"""
Verification Script for Chrome Browser Extension (Manifest V3)
==============================================================
Validates:
1. Manifest V3 syntax, required permissions, and configuration.
2. Extension icon assets (dimensions, PNG format).
3. Service worker (background.js) and Popup assets (HTML, CSS, JS).
4. Live / Mock API integration with CORS chrome-extension:// headers.
5. Fast-path and Detailed-path payload compatibility.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

from api_server.main import app

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_verification():
    print("=" * 80)
    print("VERIFYING CHROME BROWSER EXTENSION (MANIFEST V3)")
    print("=" * 80)

    ext_dir = Path("extension")
    assert ext_dir.exists(), "extension/ directory does not exist!"

    # 1. Validate manifest.json
    print("\n--- 1. Manifest V3 Schema & Configuration Validation ---")
    manifest_path = ext_dir / "manifest.json"
    assert manifest_path.exists(), "manifest.json missing!"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest.get("manifest_version") == 3, f"Expected manifest_version 3, got {manifest.get('manifest_version')}"
    assert manifest.get("name") == "Unified Phishing Detector", f"Unexpected name: {manifest.get('name')}"
    assert "version" in manifest, "Version missing in manifest"

    permissions = manifest.get("permissions", [])
    assert "activeTab" in permissions, "'activeTab' permission missing"
    assert "storage" in permissions, "'storage' permission missing"

    host_permissions = manifest.get("host_permissions", [])
    assert any("127.0.0.1:8000" in p for p in host_permissions), "127.0.0.1:8000 host permission missing"

    background = manifest.get("background", {})
    assert background.get("service_worker") == "background.js", "Background service worker missing or incorrect"

    action = manifest.get("action", {})
    assert action.get("default_popup") == "popup.html", "Default popup missing"

    print("  [PASS] manifest_version: 3 (Manifest V3)")
    print(f"  [PASS] permissions: {permissions}")
    print(f"  [PASS] host_permissions: {host_permissions}")
    print(f"  [PASS] service_worker: {background.get('service_worker')}")
    print(f"  [PASS] default_popup: {action.get('default_popup')}")

    # 2. Validate Icon Assets
    print("\n--- 2. Icon Asset Validation ---")
    icons_dir = ext_dir / "icons"
    assert icons_dir.exists(), "extension/icons/ missing!"

    required_sizes = [16, 48, 128]
    for size in required_sizes:
        icon_file = icons_dir / f"icon{size}.png"
        assert icon_file.exists(), f"Icon file {icon_file} missing!"
        with Image.open(icon_file) as img:
            assert img.format == "PNG", f"Expected PNG format for {icon_file}, got {img.format}"
            assert img.size == (size, size), f"Expected size ({size}, {size}), got {img.size}"
            print(f"  [PASS] {icon_file.name}: Valid PNG ({img.size[0]}x{img.size[1]}px, RGBA)")

    # 3. Validate HTML, CSS, and JS Files
    print("\n--- 3. Extension Source Files Integrity ---")
    files_to_check = ["background.js", "popup.html", "popup.css", "popup.js"]
    for fname in files_to_check:
        fpath = ext_dir / fname
        assert fpath.exists(), f"File {fname} missing!"
        size = fpath.stat().st_size
        assert size > 50, f"File {fname} is suspiciously small ({size} bytes)"
        print(f"  [PASS] {fname} ({size} bytes) present and verified.")

    # 4. End-to-End API Integration Simulation (CORS & Payload Verification)
    print("\n--- 4. Local API Server Integration & CORS Origin Check ---")
    with TestClient(app) as client:
        # Health check
        health_res = client.get("/health", headers={"Origin": "chrome-extension://hgnkcdkbaipmfkmffomjepffk"})
        assert health_res.status_code == 200, f"Health check failed: {health_res.text}"
        assert health_res.headers.get("access-control-allow-origin") == "chrome-extension://hgnkcdkbaipmfkmffomjepffk", "CORS header missing for chrome-extension origin"
        print("  [PASS] GET /health from chrome-extension:// origin: 200 OK + CORS Allowed")

        # Fast-path analysis simulation (Navigation ping)
        phish_url = "http://192.168.1.100/paypal-security-update/login.php?redirect=bit.ly/secure-auth"
        fast_res = client.post(
            "/analyze-url",
            headers={"Origin": "chrome-extension://hgnkcdkbaipmfkmffomjepffk"},
            json={
                "url": phish_url,
                "check_threat_intel": False,
                "generate_explanation": False,
            },
        )
        assert fast_res.status_code == 200, f"Fast path failed: {fast_res.text}"
        fast_data = fast_res.json()
        assert fast_data["verdict"] == "phishing", f"Expected phishing verdict, got {fast_data['verdict']}"
        assert fast_data["threat_intel"] is None, "Threat intel should be None on fast path"
        assert fast_data["llm_explanation"] is None, "LLM explanation should be None on fast path"
        print(f"  [PASS] Fast-path ML analysis: Verdict='{fast_data['verdict']}', Confidence={fast_data['confidence']:.1%}")
        print(f"         Top Signals: {fast_data['top_features']}")

        # Slow-path detailed explanation simulation (User button click)
        safe_url = "https://www.google.com"
        detailed_res = client.post(
            "/analyze-url",
            headers={"Origin": "chrome-extension://hgnkcdkbaipmfkmffomjepffk"},
            json={
                "url": safe_url,
                "check_threat_intel": True,
                "generate_explanation": True,
            },
        )
        assert detailed_res.status_code == 200, f"Detailed path failed: {detailed_res.text}"
        detailed_data = detailed_res.json()
        assert detailed_data["verdict"] == "safe", f"Expected safe verdict, got {detailed_data['verdict']}"
        assert detailed_data["threat_intel"] is not None, "Threat intel missing on detailed path"
        assert detailed_data["llm_explanation"] is not None, "LLM explanation missing on detailed path"
        print(f"  [PASS] Detailed-path analysis: Verdict='{detailed_data['verdict']}', Action='{detailed_data['llm_explanation']['recommended_action']}'")
        print(f"         LLM Summary: \"{detailed_data['llm_explanation']['explanation'][:80]}...\"")

    print("\n" + "=" * 80)
    print("CHROME EXTENSION VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
