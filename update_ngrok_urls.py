#!/usr/bin/env python3
"""
NGrok URL Updater for GPT Actions
Automatically updates the GPT Actions JSON with your actual ngrok tunnel URLs
"""
import json
import sys
import re

def update_ngrok_urls(json_file, api_url, processor_url, dashboard_url):
    """Update the JSON file with actual ngrok URLs"""

    try:
        # Load the JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Update server URLs
        config['servers'] = [
            {
                "url": api_url,
                "description": "ZanAnalytics API Server (Port 5010)"
            },
            {
                "url": processor_url,
                "description": "Data Processor Server (Port 5011)"
            },
            {
                "url": dashboard_url,
                "description": "Streamlit Dashboard (Port 8501)"
            }
        ]

        # Update individual endpoint servers
        endpoints_to_api = [
            "/api/market/overview",
            "/api/currency/{pair}/analysis", 
            "/api/currency/{pair}/signals",
            "/api/scanner/opportunities",
            "/api/risk/assessment",
            "/api/data/export"
        ]

        for endpoint in endpoints_to_api:
            if endpoint in config['paths']:
                for method in config['paths'][endpoint]:
                    config['paths'][endpoint][method]['servers'] = [{"url": api_url}]

        # Update data processor endpoint
        if "/process/data" in config['paths']:
            for method in config['paths']["/process/data"]:
                config['paths']["/process/data"][method]['servers'] = [{"url": processor_url}]

        # Update dashboard endpoint
        if "/dashboard/status" in config['paths']:
            for method in config['paths']["/dashboard/status"]:
                config['paths']["/dashboard/status"][method]['servers'] = [{"url": dashboard_url}]

        # Save updated JSON
        output_file = json_file.replace('.json', '_UPDATED.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"✅ Successfully updated URLs in {output_file}")
        print(f"🔌 API URL: {api_url}")
        print(f"📊 Processor URL: {processor_url}")
        print(f"📈 Dashboard URL: {dashboard_url}")

        return True

    except Exception as e:
        print(f"❌ Error updating URLs: {e}")
        return False

def validate_ngrok_url(url):
    """Validate that URL is a proper ngrok HTTPS URL"""
    pattern = r'^https://[a-zA-Z0-9\-]+\.ngrok\.io$'
    return re.match(pattern, url) is not None

if __name__ == "__main__":
    print("🚇 NGrok URL Updater for GPT Actions")
    print("="*50)

    if len(sys.argv) == 4:
        # Command line usage
        api_url = sys.argv[1]
        processor_url = sys.argv[2] 
        dashboard_url = sys.argv[3]
        json_file = 'custom_gpt_multicurrency_actions_NGROK.json'
    else:
        # Interactive usage
        print("Enter your ngrok tunnel URLs (copy from ngrok status):")
        print()

        api_url = input("🔌 ZanAnalytics API tunnel (port 5010): https://").strip()
        if not api_url.startswith('https://'):
            api_url = f"https://{api_url}"

        processor_url = input("📊 Data Processor tunnel (port 5011): https://").strip()
        if not processor_url.startswith('https://'):
            processor_url = f"https://{processor_url}"

        dashboard_url = input("📈 Streamlit Dashboard tunnel (port 8501): https://").strip()
        if not dashboard_url.startswith('https://'):
            dashboard_url = f"https://{dashboard_url}"

        json_file = 'custom_gpt_multicurrency_actions_NGROK.json'

    # Validate URLs
    urls = [api_url, processor_url, dashboard_url]
    names = ["API", "Processor", "Dashboard"]

    print("\n🔍 Validating URLs...")
    all_valid = True
    for url, name in zip(urls, names):
        if validate_ngrok_url(url):
            print(f"✅ {name}: {url}")
        else:
            print(f"❌ {name}: {url} (Invalid ngrok URL format)")
            all_valid = False

    if not all_valid:
        print("\n❌ Please check your URLs and try again.")
        sys.exit(1)

    # Update the JSON file
    print("\n🔄 Updating JSON configuration...")
    success = update_ngrok_urls(json_file, api_url, processor_url, dashboard_url)

    if success:
        print("\n🎉 Configuration updated successfully!")
        print("\n📋 Next steps:")
        print("1. Upload the *_UPDATED.json file to your GPT Actions")
        print("2. Test the API endpoints")
        print("3. Your GPT is ready to connect to your trading system!")
    else:
        print("\n❌ Failed to update configuration.")
        sys.exit(1)
