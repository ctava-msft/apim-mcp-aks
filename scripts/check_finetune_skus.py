import subprocess, json, sys

result = subprocess.run(
    ['az', 'cognitiveservices', 'account', 'list-models',
     '--name', 'cog-3aa6nbg5qypnk',
     '--resource-group', 'rg-apim-mcp-aks-2', '-o', 'json'],
    capture_output=True, text=True, shell=True
)
models = json.loads(result.stdout)

# Show all gpt-4o-mini entries
print("=== All gpt-4o-mini entries ===")
for m in models:
    name = m.get('name', '')
    if name == 'gpt-4o-mini':
        ver = m.get('model', {}).get('version', '')
        caps = m.get('capabilities', {})
        has_ft = 'FineTuneTokensMaxValue' in caps
        sku_names = [s.get('name', '') for s in m.get('skus', [])]
        kind = m.get('kind', '')
        print(f"  name={name} ver={ver} kind={kind} ft={has_ft} skus={sku_names}")

# Show fine-tunable entry details
print("\n=== Fine-tunable gpt-4o-mini details ===")
for m in models:
    if m.get('name') == 'gpt-4o-mini' and 'FineTune' in json.dumps(m.get('capabilities', {})):
        print(json.dumps(m, indent=2)[:2000])
        break
