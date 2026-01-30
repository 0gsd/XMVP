import json, sys

def count_lines(xml_path):
    print(f"Reading {xml_path}...")
    with open(xml_path, 'r') as f:
        content = f.read()
    
    # Simple extraction of Portions JSON
    p_start = content.find("<Portions>")
    p_end = content.find("</Portions>")
    
    if p_start == -1:
        print("No Portions found")
        return
        
    raw_json = content[p_start + len("<Portions>"):p_end]
    portions = json.loads(raw_json)
    
    total_lines = 0
    for p in portions:
        if "dialogue" in p:
            total_lines += len(p["dialogue"])
            
    print(f"Total Dialogue Lines: {total_lines}")

if __name__ == "__main__":
    count_lines("z_training_data/example_parodies/TheMargin_20260128_1949.xml")
