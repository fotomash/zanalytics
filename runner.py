import yaml
import os

def load_profiles(manifest_path="profiles.yaml"):
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)
    for p in manifest['profiles']:
        agent_path = p['path']
        with open(agent_path) as af:
            agent = yaml.safe_load(af)
            print(f"Loaded agent: {agent.get('profile_name', agent_path)}")
    print("All agent profiles loaded and ready.")

if __name__ == "__main__":
    load_profiles()