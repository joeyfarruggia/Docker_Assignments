import yaml
from pathlib import Path

file_path = Path("data/sample_data.yaml")
with file_path.open("r", encoding="utf-8") as file:
    data = yaml.safe_load(file)

print("Loaded YAML data:")
print(data)
print("\nStudent name:", data["student"]["name"])
print("Skills:", ", ".join(data["student"]["skills"]))
